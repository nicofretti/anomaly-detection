#ifndef _?NEW_NODE_
#define _?NEW_NODE_

#include <?pkg_name/?parent_node.h>

// Insert here general includes...

// Insert here msg and srv includes...

class ?NewNode : public ?ParentNode
{
public:
  ?NewNode(ros::NodeHandle h);
  virtual ~?NewNode();

protected:
  /*** RComponent stuff ***/

  //! Setups all the ROS' stuff
  virtual int rosSetup();
  //! Shutdowns all the ROS' stuff
  virtual int rosShutdown();
  //! Reads data a publish several info into different topics
  virtual void rosPublish();
  //! Reads params from params server
  virtual void rosReadParams();
  //! Actions performed on init state
  virtual void initState();
  //! Actions performed on standby state
  virtual void standbyState();
  //! Actions performed on ready state
  virtual void readyState();
  //! Actions performed on the emergency state
  virtual void emergencyState();
  //! Actions performed on Failure state
  virtual void failureState();

  /* RComponent stuff !*/

  /* ROS Stuff */

  //! Publishers...

  //! Subscribers...

  //! Services...

  //! Callbacks...

  /* ROS stuff !*/

  /* ?NewNode stuff */

  //! ?NewNode variables and methods...

  /* ?NewNode stuff !*/


};

#endif  // _?NEW_NODE_
