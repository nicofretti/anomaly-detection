#!/usr/bin/env python


from rcomponent.rcomponent import *

#!/usr/bin/env python

import rospy
import yaml
import rospkg
import tf
import os
from poi_manager_msgs.msg import *
from poi_manager_msgs.srv import *

from geometry_msgs.msg import Pose
from std_msgs.msg import Empty
from visualization_msgs.msg import MarkerArray, Marker



class PoiManager(RComponent):

    def __init__(self):
        self.pose_list = []
        self.pose_dict = {'environments':{}}
        RComponent.__init__(self)

    def ros_read_params(self):
        """Gets params from param server"""
        RComponent.ros_read_params(self)

        rospack = rospkg.RosPack()
        self.filename = rospy.get_param('~filename', 'test')
        self.folder = rospy.get_param('~folder', os.path.join(rospack.get_path('poi_manager'), 'config'))
        self.yaml_path = self.folder + '/' + self.filename+'.yaml'

        self.publish_markers = rospy.get_param('~publish_markers', False)

        rospy.loginfo('%s::_init_: config file path: %s', self._node_name, self.yaml_path)

    def ros_setup(self):
        """Creates and inits ROS components"""
        RComponent.ros_setup(self)

        self.service_read_yaml = rospy.Service('~read_pois', ReadPOIs, self.read_pois_cb)
        self.service_add_pois= rospy.Service('~add_pois', AddPOIs, self.add_pois_cb)
        self.service_get_poi = rospy.Service('~get_poi', GetPOI, self.get_poi_cb)
        self.service_get_poi = rospy.Service('~get_poi_params', GetPOI_params, self.get_poi_params_cb)
        self.service_delete_poi = rospy.Service('~delete_poi', DeletePOI, self.delete_poi_cb)
        self.service_delete_envir = rospy.Service('~delete_environment', DeleteEnvironment, self.delete_environment_cb)
        self.service_get_envir = rospy.Service('~get_environments', GetEnvironments, self.get_environments_cb)
        self.service_get_poi_list = rospy.Service('~get_poi_list', GetPOIs, self.get_poi_list_cb)
        self.service_add_poi = rospy.Service('~add_poi', AddPOI, self.add_poi_cb)
        self.service_add_poi_by_params = rospy.Service('~add_poi_by_params', AddPOI_params, self.add_poi_by_params_cb)

        if self.publish_markers:
            self.marker_array = MarkerArray()

            self.marker_array_publisher = rospy.Publisher('~markers', MarkerArray, queue_size=10)

    def ros_publish(self):
        '''
                Publish topics at standard frequency
        '''

        if self.publish_markers:
            self.marker_array_publisher.publish(self.marker_array)

    def parse_yaml(self):
        self.pose_dict = {'environments':{}}
        try:
            f = open(self.yaml_path, 'r')
            self.pose_dict = yaml.safe_load(f)
            if self.pose_dict is None:
                self.pose_dict = {'environments':{}}
            f.close()
            return True,"OK"
        except (IOError, yaml.YAMLError) as e:
            rospy.logerr('%s::parse_yaml: %s',rospy.get_name(),str(e))
            return False , "Error reading yaml: %s" % e

    def process_pose_dictionary(self):
        '''
            Processes the dictionary containing all the the environments and poses
            @return True if OK
            @return False if error in format
        '''
        self.pose_list = []
        envir_n = 0
        points_n = 0
        msg =""
        try:
            for key,value in self.pose_dict.items():
                # Dictionaries of point list
                for key_list,value_list in value.items():
                    # Dictionary of point list
                    envir_n = envir_n + 1
                    for point_list_key,value_point_list in value_list.items():
                        if point_list_key=='points':
                            for name_point_key,values_point in value_point_list.items():
                                points_n = points_n + 1
                                labeled_point = LabeledPose()
                                labeled_point.name = name_point_key
                                labeled_point.environment = key_list
                                for point_params_key,value_params in values_point.items():
                                    if (point_params_key=='frame_id'):
                                        labeled_point.frame_id = value_params
                                    if (point_params_key=='name'):
                                        labeled_point.name = value_params
                                    if (point_params_key=='params'):
                                        labeled_point.params = value_params
                                    if (point_params_key=='position'):
                                        labeled_point.pose.position.x=value_params[0]
                                        labeled_point.pose.position.y=value_params[1]
                                        labeled_point.pose.position.z=value_params[2]
                                    if (point_params_key=='orientation'):
                                        labeled_point.pose.orientation.x=value_params[0]
                                        labeled_point.pose.orientation.y=value_params[1]
                                        labeled_point.pose.orientation.z=value_params[2]
                                        labeled_point.pose.orientation.w=value_params[3]

                                self.pose_list.append(labeled_point)
                        else:
                            rospy.loginfo("%s::process_pose_dictionary: other property in point list: %s", rospy.get_name(),point_list_key)
        except Exception as identifier:
            msg = "Error processing dictionary containing all the environments: %s" % (identifier)
            rospy.logerr('%s::process_pose_dictionary: %s',rospy.get_name(), msg)
            return False,msg
        msg = "Read %d environments and %d points" % (envir_n,points_n)

        if self.publish_markers:
            self.update_marker_array()

        return True,msg

    def init_state(self):
        req = ReadPOIsRequest()
        self.read_pois_cb(req)
        self.switch_to_state(State.READY_STATE)

    #def ready_state(self):

        #if self.publish_markers:
        #    self.update_marker_array()

    def update_yaml(self):
        yaml_file = file(self.yaml_path, 'w')
        yaml.dump(self.pose_dict, yaml_file)
        if self.publish_markers:
            self.update_marker_array()

    def update_marker_array(self):
        marker_array = MarkerArray()
        for item in self.pose_list:
            # Two markers are created for each item
            # The first one shows the pose
            marker_arrow = self.create_marker(item, 'arrow')
            # The second one shows the label
            marker_text = self.create_marker(item, 'text')

            marker_array.markers.append(marker_arrow)
            marker_array.markers.append(marker_text)
        self.marker_array = marker_array
        self.marker_array_publisher.publish(self.marker_array)

    def ready_state(self):
        if self.publish_markers:
            self.update_marker_array()

    def create_marker(self, point, marker_type):
        marker = Marker()
        marker.header.frame_id = point.frame_id
        marker.header.stamp = rospy.Time()
        marker.ns = point.name + "_" + point.environment
        marker.action = Marker.ADD
        marker.pose.position.x = point.pose.position.x
        marker.pose.position.y = point.pose.position.y
        #quaternion = tf.transformations.quaternion_from_euler(0, 0, data[1][2])
        marker.pose.orientation.x = point.pose.orientation.x
        marker.pose.orientation.y = point.pose.orientation.y
        marker.pose.orientation.z = point.pose.orientation.z
        marker.pose.orientation.w = point.pose.orientation.w

        # TODO: parameterize arrow color
        marker.color.a = 1.0
        marker.color.g = 1.0
        marker.lifetime = rospy.Duration(1)
        if marker_type == 'arrow':
            marker.scale.x = 1.0
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.id = 0
            marker.type = Marker.ARROW
        else:
            marker.scale.z = 0.5
            marker.id = 1
            marker.type = Marker.TEXT_VIEW_FACING
            marker.text = point.name + "_" + point.environment
            marker.pose.position.z = 0.5
            marker.color.r = 1.0
            marker.color.b = 1.0
        return marker

    def read_pois_cb(self, req):
        success,msg = self.parse_yaml()
        if success == False:
            rospy.logerr('%s::read_pois_cb: Error parsing yaml file -> %s', self._node_name, msg)
            return ReadPOIsResponse(success,msg,self.pose_list)
        success,msg = self.process_pose_dictionary()
        if success == False:
            rospy.logerr('%s::read_pois_cb: Error processing data from yaml file -> %s', self._node_name, msg)
        rospy.loginfo("%s::read_pois_cb: OK", self._node_name)
        return ReadPOIsResponse(success,msg,self.pose_list)

    def add_pois_cb(self, req):
        response = AddPOIsResponse()
        req_add_poi=AddPOIRequest()
        for i in req.pose_list:
            if(i.environment != ""):
              req_add_poi.p = i
              ret = self.add_poi_cb(req_add_poi)
              if ret.success == False:
                  response.success = False
                  response.message = ret.message
                  return response

        response.success = True
        response.message = "OK"
        rospy.loginfo("%s::add_pois_cb: Read pois service done", self._node_name)
        return response


    def get_poi_cb(self, req):
        response = GetPOIResponse()
        if( req.environment == ""):
            response.success = True
            response.message = "The environment is empty, this enviroment has 0 points"
            return response
        if len(self.pose_list) > 0:
            for poi in self.pose_list:
                if poi.name == req.name and poi.environment == req.environment:
                    response.success = True
                    response.message = " Poi %s/%s found" % (req.name,req.environment)
                    response.p = poi
                    return response
        else:
            response.success = False
            response.message = " Poi %s/%s Not found, empty list" % (req.name,req.environment)
            return response
        response.success = False
        response.message = " Poi %s/%s Not found" % (req.name,req.environment)
        return response

    def get_poi_params_cb(self, req):
        response = GetPOI_paramsResponse()
        if len(self.pose_list) > 0:
            for poi in self.pose_list:
                if poi.name == req.name and poi.environment == req.environment:
                    response.success = True
                    response.message = " Poi %s/%s found" % (req.name,req.environment)
                    response.name = poi.name
                    response.environment = poi.environment
                    response.frame_id = poi.frame_id
                    response.params = poi.params
                    response.x = poi.pose.position.x
                    response.y = poi.pose.position.y
                    response.z = poi.pose.position.z
                    q = [poi.pose.orientation.x,poi.pose.orientation.y,poi.pose.orientation.z,poi.pose.orientation.w ]
                    euler = tf.transformations.euler_from_quaternion(q)
                    response.roll = euler[0]
                    response.pitch = euler[1]
                    response.yaw = euler[2]
                    return response
        else:
            response.success = False
            response.message = " Poi %s/%s Not found, empty list" % (req.name,req.environment)
            return response
        response.success = False
        response.message = " Poi %s/%s Not found" % (req.name,req.environment)
        return response

    def get_environments_cb(self, req):
        response = GetEnvironmentsResponse()
        for key,value in self.pose_dict['environments'].items():
            response.environments.append(key)
        return response

    def get_poi_list_cb(self, req):
        response = GetPOIsResponse()
        num = 0
        if len(self.pose_list) > 0:
            for poi in self.pose_list:
                if poi.environment == req.environment and req.environment!="":
                    response.p_list.append(poi)
                    num = num + 1
            if num>=0:
                response.success = True
                response.message = "  Found %d POIs from %s " % (num,req.environment)
            else:
                response.success = False
                response.message = "  Found %d POIs from %s " % (num,req.environment)
        else:
            response.success = False
            response.message = " Pois from %s Not found, empty list" % (req.environment)
        return response

    def try_create_env(self,dict_name,new_env):
        try:
            if len(dict_name['environments'][new_env])==0:
                return
        except:
           dict_name['environments'][new_env] = {}
           dict_name['environments'][new_env]['points'] = {}
        return

    def try_create_point(self,dict_name,new_env,point_name):
        try:
            if len(dict_name['environments'][new_env]['points'][point_name])==0:
                return
        except:
           dict_name['environments'][new_env]['points'][point_name] = {}
        return

    def add_poi_by_params_cb(self,req):
        if req.environment == "":
            response = AddPOI_paramsResponse()
            response.success = False
            response.message = "The environment is empty"
            return response

        p = LabeledPose()
        p.name = req.name
        p.frame_id = req.frame_id
        p.environment = req.environment
        p.params = req.params

        quaternion = tf.transformations.quaternion_from_euler(req.roll,req.pitch,req.yaw)
        p.pose.position.x=req.x
        p.pose.position.y=req.y
        p.pose.position.z=req.z
        #print (quaternion)
        p.pose.orientation.x = quaternion[0]
        p.pose.orientation.y = quaternion[1]
        p.pose.orientation.z = quaternion[2]
        p.pose.orientation.w = quaternion[3]

        req_add_poi = AddPOIRequest()
        req_add_poi.p = p

        ret = self.add_poi_cb(req_add_poi)

        response = AddPOI_paramsResponse()

        response.success = ret.success
        response.message = ret.message

        return response

    def delete_environment_cb(self,req):
        response = DeleteEnvironmentResponse()
        if req.environment == "":
            response.success = False
            response.message = "The environment is empty"
            return response
        try:
            del (self.pose_dict['environments'][req.environment]['points'])
            del (self.pose_dict['environments'][req.environment])
            yaml_file = file(self.yaml_path, 'w')
            yaml.dump(self.pose_dict, yaml_file)
            response.success = True
            response.message = "Environment %s deleted" % (req.environment )
            #print (response.message)

        except Exception as identifier:
            msg = "%s::Error deleting environment %s. Error msg:%s" % (rospy.get_name(),req.environment,identifier)
            response.success = False
            response.message = msg
            #print (response.message)
        return response



    def delete_empty_environment(self,envir):
        try:
            if len(self.pose_dict['environments'][envir]['points'])==0:
                del (self.pose_dict['environments'][envir]['points'])
                del (self.pose_dict['environments'][envir])
        except:
            return
        return


    def delete_poi_cb(self,req):
        response = DeletePOIResponse()
        try:
            del (self.pose_dict['environments'][req.environment]['points'][req.name])

            self.delete_empty_environment(req.environment)
            response.success = True
            response.message = "point %s from environment %s deleted" % (req.name,req.environment )
            yaml_file = file(self.yaml_path, 'w')
            yaml.dump(self.pose_dict, yaml_file)
        except Exception as identifier:
            msg = "%s::Error deleting point %s from environment %s. Error msg:%s" % (rospy.get_name(),req.name,req.environment,identifier)
            response.success = False
            response.message = msg
        return response

    def add_poi_cb(self,req):
        response = AddPOIResponse()
        if req.p.environment == "":
            response.message = "The environment is empty"
            response.success = False
            return response
        try:
            self.try_create_env(self.pose_dict,req.p.environment)
            self.try_create_point(self.pose_dict,req.p.environment,req.p.name)
            point  = {'position':[float(req.p.pose.position.x),
                                  float(req.p.pose.position.y),
                                  float(req.p.pose.position.z)],
                    'orientation':[ float(req.p.pose.orientation.x),
                                    float(req.p.pose.orientation.y),
                                    float(req.p.pose.orientation.z),
                                    float(req.p.pose.orientation.w)],
                    'frame_id':req.p.frame_id,
                    'params':req.p.params}
            self.pose_dict['environments'][req.p.environment]['points'][req.p.name] = point

            self.pose_list.append(req.p)

            #print (self.pose_list)

            success,msg=self.process_pose_dictionary()
            yaml_file = file(self.yaml_path, 'w')
            yaml.dump(self.pose_dict, yaml_file, default_flow_style=False)

            if success == False:
                response.success = False
                response.message = "point %s from environment %s Not created/modified ERROR adding to pose list" % (req.p.name,req.p.environment )
            response.success = True
            response.message = "point %s from environment %s created/modified" % (req.p.name,req.p.environment )


        except Exception as identifier:
            msg = "Error adding point %s into environment %s: %s" % (req.p.name,req.p.environment,identifier)
            rospy.logerr('%s::add_poi_cb: %s', rospy.get_name(), msg)
            response.success = False
            response.message = msg

        return response
