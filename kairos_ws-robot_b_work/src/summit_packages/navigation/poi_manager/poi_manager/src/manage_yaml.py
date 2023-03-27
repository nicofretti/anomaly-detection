#!/usr/bin/env python

import rospy
import yaml
import rospkg
import tf
import os
from poi_manager.msg import *
from poi_manager.srv import *
from geometry_msgs.msg import Pose2D
from std_msgs.msg import Empty
from visualization_msgs.msg import MarkerArray, Marker
from robotnik_msgs.srv import GetPOI, GetPOIResponse


class PoiManager:

    def __init__(self):
        rospack = rospkg.RosPack()
        self.filename = rospy.get_param('~filename', 'test')
        self.folder = rospy.get_param('~folder', os.path.join(rospack.get_path('poi_manager'), 'config'))
        self.yaml_path = self.folder + '/' + self.filename+'.yaml'

        self.pose_list = []
        self.pose_dict = {}
        self.service_read_yaml = rospy.Service('~read_pois', ReadPOIs, self.handle_labeled_pose_list)
        self.service_write_data = rospy.Service('~update_pois', UpdatePOIs, self.handle_updated_list)
        self.service_get_poi = rospy.Service('~get_poi', GetPOI, self.get_poi_cb)

        self.publish_markers = rospy.get_param('~publish_markers', False)
        if self.publish_markers:
            self.marker_array = MarkerArray()
            self.marker_topic = rospy.get_param('~marker_topic', 'markers')
            self.frequency = rospy.get_param('~frequency', 0.5)
            self.frame_id = rospy.get_param('~frame_id', 'map')
            self.marker_array_publisher = rospy.Publisher(
                self.filename+'/'+self.marker_topic, MarkerArray, queue_size=10)

        rospy.loginfo('%s::_init_: config file path: %s', rospy.get_name(), self.yaml_path)

    def start(self):
        while not rospy.is_shutdown() and self.publish_markers:
            self.marker_array_publisher.publish(self.marker_array)
            rospy.sleep(1/self.frequency)

    def parse_yaml(self):
        try:
            f = open(self.yaml_path, 'r')
            self.pose_dict = {}
            self.pose_dict = yaml.safe_load(f)
            if self.pose_dict is None:
                self.pose_dict = {}
            f.close()
        except (IOError, yaml.YAMLError) as e:
            rospy.logerr(e)
            return 0

    def manage_read_data(self):
        self.pose_list = []
        for key, value in self.pose_dict.items():
            self.pose_list.append(LabeledPose(key, Pose2D(value[0], value[1], value[2])))

        if self.publish_markers:
            self.update_marker_array()

    def update_yaml(self, req):
        yaml_file = file(self.yaml_path, 'w')
        pose_list = req.pose_list
        self.pose_dict = {}
        for elem in pose_list:
            self.pose_dict[elem.label] = [elem.pose.x, elem.pose.y, elem.pose.theta]
        yaml.dump(self.pose_dict, yaml_file)

        if self.publish_markers:
            self.update_marker_array()

    def update_marker_array(self):
        marker_array = MarkerArray()
        for item in self.pose_dict.items():
            # Two markers are created for each item
            # The first one shows the pose
            marker_arrow = self.create_marker(item, 'arrow')
            # The second one shows the label
            marker_text = self.create_marker(item, 'text')
            marker_array.markers.append(marker_arrow)
            marker_array.markers.append(marker_text)
        self.marker_array = marker_array
        self.marker_array_publisher.publish(self.marker_array)

    def create_marker(self, data, marker_type):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time()
        marker.ns = data[0]
        marker.action = Marker.ADD
        marker.pose.position.x = data[1][0]
        marker.pose.position.y = data[1][1]
        quaternion = tf.transformations.quaternion_from_euler(0, 0, data[1][2])
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]
        marker.scale.x = 1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        # TODO parameterize arrow color
        marker.color.a = 1.0
        marker.color.g = 1.0
        marker.lifetime = rospy.Duration(1/self.frequency)
        if marker_type == 'arrow':
            marker.id = 0
            marker.type = Marker.ARROW
        else:
            marker.id = 1
            marker.type = Marker.TEXT_VIEW_FACING
            marker.text = data[0]
            marker.pose.position.z = 0.5
            marker.color.r = 1.0
            marker.color.b = 1.0
        return marker

    def handle_labeled_pose_list(self, req):
        self.parse_yaml()
        self.manage_read_data()
        rospy.loginfo("%s::handle_labeled_pose_list: read_pois service done", rospy.get_name())
        return ReadPOIsResponse(self.pose_list)

    def handle_updated_list(self, req):
        self.update_yaml(req)
        rospy.loginfo("%s::handle_updated_list: update_pois service done", rospy.get_name())
        return UpdatePOIsResponse()

    def get_poi_cb(self, req):
        name = req.name
        response = GetPOIResponse()
        if len(self.pose_list) > 0:
            for poi in self.pose_list:
                if poi.label == name:
                    response.success = True
                    response.pose = poi.pose
                    return response
        else:
            response.success = False

        return response


def main():
    rospy.init_node('poi_manager')
    yaml_manager = PoiManager()
    yaml_manager.start()
    rospy.loginfo("%s::main: spin", rospy.get_name())
    rospy.spin()


if __name__ == "__main__":
    main()
