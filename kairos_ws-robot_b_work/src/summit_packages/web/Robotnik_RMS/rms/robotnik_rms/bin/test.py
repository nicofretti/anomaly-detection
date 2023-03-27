#!/usr/bin/env python
import rospy                                   # the main module for ROS-python programs
from map_nav_manager.srv import GoToTag
from map_nav_manager.srv import GoToTagResponse # we are creating a 'Trigger service'...
                                                  # ...Other types are available, and you can create
                                                  # custom types
def trigger_response(request):
    ''' 
    Callback function used by the service server to process
    requests from clients. It returns a GoToTagResponse
    '''
    print(request)
    return GoToTagResponse(

        ret=True,
        msg="Test"
    )
 
rospy.init_node('map_nav_manager_test')                     # initialize a ROS node
my_service = rospy.Service(                        # create a service, specifying its name,
    '/poi_interactive_marker/goto_tag', GoToTag, trigger_response         # type, and callback
)
rospy.spin()                                       # Keep the program from exiting, until Ctrl + C is pressed