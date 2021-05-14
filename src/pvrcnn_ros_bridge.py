import easydict as edict
import glob
from pathlib import Path
import time
import datetime
import numpy as np
import torch

import rospy
import ros_numpy as rnp
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils.box_utils import boxes_to_corners_3d


def exetime(func):
    def newfunc(*args, **args2):
        _t0 = time.time()
        back = func(*args, **args2)
        _t1 = time.time()
        print("{:20s}".format(func.__name__) + " : " + "{:5.1f}".format((_t1 - _t0) * 1000) + "ms")
        return back

    return newfunc


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class pvrcnn_ros:
    def __init__(self, input_dict):
        # pvrcnn cfg
        cfg_from_yaml_file(input_dict.cfg_file, cfg)

        log_dir = Path(str(input_dict.output_dir)) / 'log'
        log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = common_utils.create_logger(
            log_dir / ('log_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')), rank
            =cfg.LOCAL_RANK)

        self.demo_dataset = DemoDataset(  # dummy dataset for preprocess inputdata
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=input_dict.dummy_cloud,
            ext='.bin', logger=self.logger
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.model.load_params_from_file(filename=input_dict.ckpt_file, logger=self.logger,
                                         to_cpu=self.device == "cpu")

        self.model.to(self.device)
        self.model.eval()

        # for ROS
        self.sub = rospy.Subscriber(input_dict.topic, PointCloud2, self.pc2_callback, queue_size=1)
        self.mk_pub = rospy.Publisher("ros_pvrcnn", MarkerArray, queue_size=1)
        self.frame_id = 0

    @exetime
    def detector(self, points):
        with torch.no_grad():
            input_dict = {
                'points': points,
                'frame_id': self.frame_id,
            }
            data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            pred_dicts = self.model.forward(data_dict)[0][0]  # batch_size = 1
            # print(pred_dicts['pred_boxes'].shape)
            self.viz(pred_dicts['pred_boxes'].detach().cpu().numpy(), "excavator/LiDAR_80_1")
            print(pred_dicts, cfg.CLASS_NAMES)
            # pred = self.remove_low_score(pred_dicts[0])

            # mea_data = {}
            # mea_data["time_stamp"] = pointcloud_msg.header.stamp.to_sec()
            # mea_data["pointcloud"] = self.pointcloud
            # mea_data["boxes_3d"] = pred[
            #     "pred_boxes"].detach().cpu().numpy()  # boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            # # mea_data["boxes_3d_scores"] = pred["pred_scores"].detach().cpu().numpy()
            # # mea_data["boxes_3d_types"] = pred["pred_labels"].detach().cpu().numpy()
            # self.det_3dbbox.append(mea_data)
            # self.det_pc_handling = False

    @exetime
    def pc2_callback(self, msg):
        points_raw = rnp.point_cloud2.pointcloud2_to_xyz_array(msg)
        points_raw[:, 2] = points_raw[:, 2]
        points_raw = np.hstack((points_raw, np.zeros([len(points_raw), 1])))
        self.detector(points_raw)
        self.frame_id += 1
        print('-------------------------------------')

    def viz(self, bbox3d, frame_id):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.type = marker.LINE_LIST
        marker.action = marker.ADD
        marker.header.stamp = rospy.Time.now()

        # marker scale (scale y and z not used due to being linelist)
        marker.scale.x = 0.08
        # marker color
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.points = []
        corner_for_box_list = [0, 1, 0, 3, 2, 3, 2, 1, 4, 5, 4, 7, 6, 7, 6, 5, 3, 7, 0, 4, 1, 5, 2, 6]
        corners3d = boxes_to_corners_3d(bbox3d)  # (N,8,3)
        for box_nr in range(corners3d.shape[0]):
            box3d_pts_3d_velo = corners3d[box_nr]  # (8,3)
            for corner in corner_for_box_list:
                transformed_p = np.array(box3d_pts_3d_velo[corner, 0:4])
                # transformed_p = transform_point(p, np.linalg.inv(self.Tr_velo_kitti_cam))
                p = Point()
                p.x = transformed_p[0]
                p.y = transformed_p[1]
                p.z = transformed_p[2]
                marker.points.append(p)
        marker_array.markers.append(marker)

        id = 0
        for m in marker_array.markers:
            m.id = id
            id += 1
        self.mk_pub.publish(marker_array)
        marker_array.markers = []
        pass
