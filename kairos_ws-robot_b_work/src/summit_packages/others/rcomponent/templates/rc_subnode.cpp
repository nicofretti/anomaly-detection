#include <?pkg_name/?new_node.h>

?NewNode::?NewNode(ros::NodeHandle h) : ?ParentNode(h)
{
  component_name.assign(pnh_.getNamespace());
  rosReadParams();
}

?NewNode::~?NewNode()
{
}

void ?NewNode::rosReadParams()
{
  ?ParentNode::rosReadParams();

  bool required = true;
  bool not_required = false;

  // Parameters ...
}

int ?NewNode::rosSetup()
{
  ?ParentNode::rosSetup();

  bool required = true;
  bool not_required = false;

  // Publishers ...

  // Subscribers ...

  // Services ...
  
}

int ?NewNode::rosShutdown()
{
  ?ParentNode::rosShutdown();

  // Shutsown state...
}

void ?NewNode::rosPublish()
{
  ?ParentNode::rosPublish();
  
  // Ros publish state...
}

void ?NewNode::initState()
{
  ?ParentNode::initState();

  // Init state...
}

void ?NewNode::standbyState()
{
  ?ParentNode::standbyState();

  // Standby state...
}

void ?NewNode::readyState()
{
  ?ParentNode::readyState();

  // Ready state...
}

void ?NewNode::emergencyState()
{
  ?ParentNode::emergencyState();

  // Emergency state...
}

void ?NewNode::failureState()
{
  ?ParentNode::failureState();

  // Failure state...
}