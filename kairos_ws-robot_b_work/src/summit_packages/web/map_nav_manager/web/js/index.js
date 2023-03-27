  /**
   * Setup all visualization elements when the page is loaded. 
   */
  function init() {
    // Connect to ROS.
    var ros = new ROSLIB.Ros({
      url : 'ws://'+hostname+':9090'
    });

    
  }
