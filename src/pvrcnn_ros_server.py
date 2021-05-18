#!/home/ou/software/anaconda3/envs/pvrcnn/bin/python
from pathlib import Path
from easydict import EasyDict

import rospy
import rospkg

from pvrcnn_ros_bridge import pvrcnn_ros

root_path = Path(rospkg.RosPack().get_path("ros_pvrcnn"))

if __name__ == "__main__":
    rospy.init_node('ros_pvrcnn_node')

    input_dict = EasyDict()
    # input_dict.output_dir = root_path / "output"
    # input_dict.cfg_file = root_path / "config/ironpile/stone_pv_rcnn.yaml"
    # input_dict.ckpt_file = root_path / "config/ironpile/checkpoint_epoch_80.pth"
    # input_dict.dummy_cloud = root_path / "config/000000.bin"

    input_dict.output_dir = root_path / "output"
    input_dict.cfg_file = root_path / "config/person_truck/person_truck_pv_rcnn.yaml"
    input_dict.ckpt_file = root_path / "config/person_truck/person_truck_checkpoint_epoch_80.pth"
    input_dict.dummy_cloud = root_path / "config/000000.bin"
    input_dict.score_threashold = 0.5
    detector = pvrcnn_ros(input_dict)

    rospy.spin()
