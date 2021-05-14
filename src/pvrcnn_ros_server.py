import rospy
import rospkg
from pathlib import Path
from easydict import EasyDict
from pvrcnn_ros_bridge import pvrcnn_ros

root_path = Path(rospkg.RosPack().get_path("ros_pvrcnn"))

if __name__ == "__main__":
    rospy.init_node('ros_pvrcnn_node')

    input_dict = EasyDict()
    input_dict.output_dir = root_path / "output"
    input_dict.cfg_file = root_path / "config/pv_rcnn.yaml"
    input_dict.ckpt_file = root_path / "config/pv_rcnn_8369.pth"
    input_dict.dummy_cloud = root_path / "config/000000.bin"
    input_dict.topic = "/excavator/LiDAR_80_1"

    detector = pvrcnn_ros(input_dict)

    rospy.spin()
