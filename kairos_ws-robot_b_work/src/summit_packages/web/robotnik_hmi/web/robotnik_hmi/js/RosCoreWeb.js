/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

var timerRefreshMap = 5000;
var timerRefreshArrayMarker = 1000;
var timerRefreshMarker = 1000;
var reloadPage = 300000;

var refreshMap = true;
var refreshArrayMarker = true;
var refreshMarker = true;


var ros = new ROSLIB.Ros();
//var ros = new ROSLIB.Ros({
//    url : 'ws://'+hostname
//});

var address = "";
var main_loop_update_time = 1000; // in ms
var global_frame = "/robot_map";
//var joystic_max_linear_vel = 0.5; // m/s
//var joystic_max_angular_vel = 0.5; // rad/s


//if you have a error with the conexion with rosbridge
ros.on('error', function (error) {
    alert("Error to connect with ROS core");
    //listener.unsubscribe();
});


//if you have connection with rosbridge
ros.on('connection', function () {
    //alert("Connect with ROS core");
});

//if you don't have conexion with the conexion with rosbridge
ros.on('close', function () {
    alert("Close conection with ROS");
    //listener.unsubscribe();
});

//Close all subcriptions
function unsubscribe() {
    listener.unsubscribe();
};

/**
 * Define a subscribe with ros
 * @param {string} name_topic Name topic
 * @param {string} message_Type Type of message
 * @param {function} mess_function Function to do when recibe a message
 */
function subscribe(name_topic, message_Type, mess_function, queque_size=1) {
    var listener = new ROSLIB.Topic({
        ros: ros,
        name: name_topic,
        messageType: message_Type,
	queue_size: queque_size
    });
    listener.subscribe(function (message) {
        mess_function(message);
    });
}


/**
 * Call a service in ros
 * @param {string} name_service Name service
 * @param {string} servie_Type Type of service
 * @param {struct} service_request A struct message will be send to ros
 * @param {struct} response_fucntion Function to do when recibe a response to ros service
 *
*/
function callService(name_service, service_Type, service_request, response_function) {
    var service = new ROSLIB.Service({
        ros: ros,
        name: name_service,
        serviceType: service_Type
    });
    var mess = new ROSLIB.ServiceRequest(service_request);
    service.callService(mess, function (result) {
        response_function(result);
    });
}


/**
 * Gets a rosparam
 * @param {string} namespace param namespace
 * @param {struct} callback Callback when receiving the response
 *
*/
function getParam(namespace,  callback) {
    // Create a Param object for the param
    var param = new ROSLIB.Param({
         ros : ros,
         name : namespace
    });
    // Get the value of the param
    param.get(function(value) {
       callback(value);
    });
}


/**
 * Publish a message in ros
 * @param {string} nametopic Topic when you will publish
 * @param {string} messageType Type of message
 * @param {strung} message A struct when you push the message
 */
function publish(nametopic, messageType, message) {
    var pub = new ROSLIB.Topic({
        ros: ros,
        name: nametopic,
        serviceType: messageType
    });
    var mes = new ROSLIB.Message(message);
    pub.publish(mes);

}

var viewer = null;

/**
 * Create a viewer
 * @param {string} _idDiv Id when do a Viewer
 */
function createViewer(_idDiv) {
    viewer = new ROS3D.Viewer({
        divID: _idDiv,
        width: 800,
        height: 800,
        antialias: true,
        background: '#002233',
        cameraPose: { x: 3, y: 3, z: 1 }
    });
    return viewer;
}

/**
 * Create a grid in viewer
 */
function addGrid() {
    if (viewer != null) {
        viewer.addObject(new ROS3D.Grid());
    } else {
        console.log("not declarate a Viewer, use function createViewer")
    }
}

var tfClient = null;

/**
 * Define where the TFs for the viewer are obtained
 * @param {string} _fixedFrame Frame
 * @param {string} _serverName TF server name
 * @param {number} _rate Refresh rate
 */
function setTfClient(_fixedFrame, _serverName, _repubServiceName, _rate = 50) {
    tfClient = new ROSLIB.TFClient({
        ros: ros,
        fixedFrame: _fixedFrame,
        angularThres: 0.01,
        transThres: 0.01,
        rate: _rate,
        serverName: _serverName,
        repubServiceName: _repubServiceName

    });
    return tfClient;
};

function setTfClient(_fixedFrame) {
    tfClient = new ROSLIB.TFClient({
        ros: ros,
        angularThres: 0.01,
        transThres: 0.01,
        rate: 10.0,
        fixedFrame: _fixedFrame
    });
    return tfClient;
}

var robostDefinitions = [];

/**
 * Define a robot definition
 * @param {string} _topicDefinition Topic definition
 */

function addRobot(_topicDefinition) {
    if (viewer == null) {
        console.log("not declarate a Viewer, use function createViewer");
        return null;
    }
    if (tfClient == null) {
        console.log("not declarate a TFClient, use function setTfClient");
        return null;
    }
    var r = new ROS3D.UrdfClient({
        ros: ros,
        param: _topicDefinition,
        path: address,
        tfClient: tfClient,
        rootObject: viewer.scene
    });
    robostDefinitions.push(r);
    return r;
}

var markersDefinitions = [];

/**
 * Define a marker
 * @param {string} _topicMarker Topic marker
 */

function addMarker(_topicMarker) {
    if (viewer == null) {
        console.log("not declarate a Viewer, use function createViewer");
        return null;
    }
    if (tfClient == null) {
        console.log("not declarate a TFClient, use function setTfClient");
        return null;
    }
    var marker = new ROS3D.MarkerClient({
        ros: ros,
        tfClient: tfClient,
        topic: _topicMarker,
        rootObject: viewer.scene
    });
    markersDefinitions.push(marker);
    return marker;
}

var arrayMarkersDefinitions = [];

/**
 * Define array marker
 * @param {string} _topicArrayMarker Topic arry marker
 */

function addArrayMarker(_topicArrayMarker) {
    if (viewer == null) {
        console.log("not declarate a Viewer, use function createViewer");
        return null;
    }
    if (tfClient == null) {
        console.log("not declarate a TFClient, use function setTfClient");
        return null;
    }
    var marker = new ROS3D.MarkerArrayClient({
        ros: ros,
        tfClient: tfClient,
        topic: _topicArrayMarker,
        rootObject: viewer.scene
    });
    arrayMarkersDefinitions.push(marker);
    return marker;
}


var gridClient = null;

/**
 * Define where ros publishs the map
 * @param {string} _topicMap Topic map
 * @param {bool} _continuos Define the refresh, if is continuos the map change, else not
 */

function setMap(_topicMap, _continuos = true) {
    if (viewer == null) {
        console.log("not declarate a Viewer, use function createViewer");
        return null;
    }
    gridClient = new ROS3D.OccupancyGridClient({
        ros: ros,
        rootObject: viewer.scene,
        continuous: _continuos,
        topic: _topicMap
    });
}

var imClient = null;

/**
 * Define a interactive marker
 * @param {sting} _topic Topic in ros
 */
function interactiveMarkerClient(_topic) {
    if (viewer == null) {
        console.log("not declarate a Viewer, use function createViewer");
        return null;
    }
    imClient = new ROS3D.InteractiveMarkerClient({
        ros: ros,
        tfClient: tfClient,
        topic: _topic,
        camera: viewer.camera,
        rootObject: viewer.selectableObjects
    });
    return imClient;
}

/**
 * Function aux to get a data in plane in form recurrent
 * @param {Ajax} data Data type Ajax
 * @param {string} nameKeyFather Name of father data
 * @param {Ajax} dataExit Data in plane
 */
function recurentData(data, nameKeyFather, dataExit) {
    Object.keys(data).forEach(function (k) {
        if (typeof data[k] == "object") {
            var aux = recurentData(data[k], nameKeyFather + ";" + k, dataExit);
            Object.keys(aux).forEach(function (k) {
                dataExit[k] = aux[k].toString();
            });
        } else {
            dataExit[nameKeyFather + ";" + k] = data[k].toString();
        }
    });
    return dataExit;
}

/**
 * Function to get all levels to data ajax to ajax plane
 * @param {Ajax} data Data to plane
 * @return {Ajax} Data in plane
 */
function planeData(data) {
    dataExit = {};
    nameKeyFather = "data";
    if (data == null) {
        return dataExit;
    }
    return dataExit = recurentData(data, nameKeyFather, dataExit);
}


/**
 *
 */
function pointCloud2(_topic) {
    if (viewer == null) {
        console.log("not declarate a Viewer, use function createViewer");
        return null;
    }
    point_cloud2 = new ROS3D.PointCloud2({
        ros: ros,
        topic: _topic,
        tfClient: tfClient,
        rootObject: viewer.scene,
        max_pts: 10000
    });
    return point_cloud2;
}

/**
 *
 */
function laserScan(_topic, _size=0.15, _color="#FF0000") {
    if (viewer == null) {
        console.log("not declarate a Viewer, use function createViewer");
        return null;
    }
    laser_scan = new ROS3D.LaserScan({
        ros: ros,
        topic: _topic,
        tfClient: tfClient,
        rootObject: viewer.scene,
        //color: 0xFF0000,
        material : { color: _color, size: _size},
        //pointRatio : 0.01
        //size: _size
    });
    return laser_scan;
}

/**
 *
 */
function pointCloud2(_topic, _size=0.15, _color="#00FF00") {
    if (viewer == null) {
        console.log("not declarate a Viewer, use function createViewer");
        return null;
    }
    point_cloud_2 = new ROS3D.PointCloud2({
        ros: ros,
        topic: _topic,
        tfClient: tfClient,
        rootObject: viewer.scene,
        material : { color: _color, size: _size},
    });
    return point_cloud_2;
}

function path(_topic, _color="#00FF00") {
    if (viewer == null) {
        console.log("not declarate a Viewer, use function createViewer");
        return null;
    }
    view_path = new ROS3D.Path({
        ros: ros,
        topic: _topic,
        tfClient: tfClient,
        rootObject: viewer.scene,
        color: _color
    });
    return view_path;
}

function pose_array(_topic, _color="#FF0000") {
    if (viewer == null) {
        console.log("not declarate a Viewer, use function createViewer");
        return null;
    }
    view_path = new ROS3D.Path({
        ros: ros,
        topic: _topic,
        tfClient: tfClient,
        rootObject: viewer.scene,
        color: _color
    });
    return view_path;
}
/*
function triggerMap (){
    refreshMap = true;
}

function triggerArrayMarker(){
    refreshArrayMarker = true;
}

function triggerMarker(){
    refreshMarker = true;
}


setInterval(triggerMap, timerRefreshMap);
setInterval(triggerArrayMarker, timerRefreshArrayMarker);
setInterval(triggerMarker, timerRefreshMarker);
setInterval(function (){
    location.reload(true);
}, reloadPage)
*/
