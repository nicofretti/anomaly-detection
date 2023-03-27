var camera_front_pan= 0.0;
var camera_front_tilt= 0.0;	
var camera_front_zoom = 0.0;
var camera_rear_pan= 0.0;
var camera_rear_tilt= 0.0;	
var camera_rear_zoom = 0.0;
var pan_max = 180.0;
var pan_min = -180.0;
var tilt_max = 0;
var tilt_min = -90.0;

var is_recording = 0;
var yaw_angle = 0.0;
var yaw_angle_left = 0.0;
var yaw_angle_right = 0.0;
var default_max_angle = 5.0;
var max_angle = default_max_angle;
var max_angle_message;
var default_trim_angle = 0.0;
var trim_angle = default_trim_angle;
var trim_angle_message;
var max_angle_defined = false;
var trim_angle_defined = false;
var battery_level = 0.0;
var battery_level_corrected;
var maximum_battery_level = 53.0;
var minimum_battery_level = 46.5;
var battery_status = false;
var min_radius = 0.0;
var reverse_direction = false;
var deactivate_imu_msg;
var imu_state = false;
var imu_temperature = 0.0;
var imu_temperature_status = 0.0;
var limit_temperature = 60.0;
var x_position;
var y_position;
var z_position;
var x_orientation;
var y_orientation;
var z_orientation;
var emergency_button_status = false;
var flw_data = ["","","",""];
var blw_data = ["","","",""];
var frw_data = ["","","",""];
var brw_data = ["","","",""];
var status_word_string = "";
var status_word_sub_string = "";

// wheel drivers flag history
var flw_data_history = ["","","",""];
var blw_data_history = ["","","",""];
var frw_data_history = ["","","",""];
var brw_data_history = ["","","",""];

//controller options
var gearbox_reduction = 12.52;
var diameter_wheel = 0.207;
var motors_encoders_factor = 4000;
var take_over= true;
var kinematic_mode = "skid";
var motion_odometry = "false";
var motors_encoder = "false";
var x_wam= false;

var status_codes = 
["SW_READY_TO_SWITCH_ON", //0
"SW_SWITCHED_ON",
"SW_OP_ENABLED",
"SW_FAULT",
"SW_VOLTAGE_ENABLED",
"SW_QUICK_STOP",		//5
"SW_SWITCH_ON_DISABLED",
"SW_WARNING",
"SW_DELAY",
"FLAG NOT USED",
"SW_TARGET_REACHED",	//10
"FLAG NOT USED",
"SW_HOME_ATTAINED",
"SW_HOMING_ERROR",
"SW_CAPTURE",		//14
"FLAG NOT USED"];

var drive_status_codes = [
"DS_BRIDGE_ENABLED",  //0
"DS_DYNAMIC_BRAKE_ENABLED",
"DS_SHUNT_ENABLED",
"DS_POSITIVE_STOP_ENABLED",
"DS_NEGATIVE_STOP_ENABLED",
"DS_POSITIVE_TORQUE_INHIBIT_ACTIVE",	//5
"DS_NEGATIVE_TORQUE_INHIBIT_ACTIVE",
"DS_EXTERNAL_BRAKE_ACTIVE",
"DS_DRIVE_RESET",
"DS_DRIVE_INTERNAL_ERROR",
"DS_SHORT_CIRCUIT",//10
"DS_CURRENT_OVERSHOOT",
"DS_UNDER_VOLTAGE",
"DS_OVER_VOLTAGE",
"DS_DRIVER_OVER_TEMPERATURE",
"DS_PARAMETER_RESTORE_ERROR",//15
"DS_PARAMETER_STORE_ERROR",
"DS_INVALID_HALL_STATE",
"DS_PHASE_SYNC_ERROR",
"DS_MOTOR_OVER_TEMPERATURE",
"DS_PHASE_DETECTION_FAULT",//20
"DS_FEEDBACK_SENSOR_ERROR",
"DS_MOTOR_OVER_SPEED",
"DS_MAX_MEASURED_POSITION",
"DS_MIN_MEASURED_POSITION",
"DS_COMMUNICATION_ERROR",//25
"DS_LOG_ENTRY_MISSED",
"DS_COMMANDED_DISABLE",
"DS_USER_DISABLE",
"DS_POSITIVE_INHIBIT",
"DS_NEGATIVE_INHIBIT",//30
"DS_CURRENT_LIMITING",
"DS_CONTINUOUS_CURRENT",
"DS_CURRENT_LOOP_SATURED",
"DS_USER_UNDER_VOLTAGE",
"DS_USER_OVER_VOLTAGE",//35
"DS_NONSINUSOIDAL_COMMUTATION",
"DS_PHASE_DETECTION",
"DS_USER_AUXILIARY_DISABLE",
"DS_SHUNT_REGULATOR",
"DS_PHASE_DETECTION_COMPLETE",//40
"DS_ZERO_VELOCITY",
"DS_AT_COMMAND",
"DS_VELOCITY_FOLLOWING_ERROR",
"DS_POSITIVE_TARGET_VELOCITY_LIMIT",
"DS_NEGATIVE_TARGET_VELOCITY_LIMIT",//45
"DS_COMMAND_LIMITER_ACTIVE",
"DS_IN_HOME_POSITION",
"DS_POSITON_FOLLOWING_ERROR",
"DS_MAX_TARGET_POSITION_LIMIT",
"DS_MIN_TARGET_POSITION_LIMIT",//50
"DS_LOAD_MEASURED_POSITION",
"DS_LOAD_TARGET",
"DS_HOMING_ACTIVE",
"DS_HOMING_COMPLETE",
"DS_PVT_BUFFER_FULL",//55
"DS_PVT_BUFFER_EMPTY",
"DS_PVT_BUFFER_THRESHOLD",
"DS_PVT_BUFFER_FAILURE",
"DS_PVT_BUFFER_EMPTY_STOP",
"DS_PVT_BUFFER_SEQUENCE_ERROR",//60
"DS_COMMANDED_STOP",
"DS_USER_QUICKSTOP",
"DS_COMMANDED_POSITIVE_LIMIT",
"DS_COMMANDED_NEGATIVE_LIMIT",
"FLAG NOT USED"]; //65



var ros = new ROSLIB.Ros({
	//url : 'ws://'+hostname+':9090'
        url : 'ws://'+hostname+':9090'
});


// Publishers
// -----------
var camera_front_ptz_publisher = new ROSLIB.Topic({  
	ros : ros,
	name : namespace+'/front_ptz_camera/front_ptz_camera/ptz_command',
	messageType : 'robotnik_msgs/ptz'
});
var camera_rear_ptz_publisher = new ROSLIB.Topic({  
	ros : ros,
	name : namespace+'/camera_rear/ptz_command',
	messageType : 'robotnik_msgs/ptz'
});


// Subscribing to Topics
// ----------------------

//Subscriber to system_monitor/diagnostics topic to set all necessary elements to his tab
var monitorListener = new ROSLIB.Topic({
    ros : ros,
    name : namespace+'/system_monitor/diagnostics',
    messageType : 'system_monitor/Diagnostic'
});

//battery topic
var battery_subscriber = new ROSLIB.Topic({
	ros : ros,
	name : namespace+'/robotnik_base_hw/battery',
	messageType : 'std_msgs/Float32'
});



//imu state topic
var imuStateListener = new ROSLIB.Topic({
	ros : ros,
	name : namespace+'/imu/data_ok',
	messageType : 'std_msgs/Bool'
});



// IMU temperature topic (Arduimu)
var imu_temperature_subscriber = new ROSLIB.Topic({
	ros : ros,
	name : namespace+'/mavros/imu/temperature',
	messageType : 'sensor_msgs/Temperature'
});

//emergency stop topic
var emergency_stop_subscriber = new ROSLIB.Topic({
	ros : ros,
	name : namespace+'/robotnik_base_hw/emergency_stop',
	messageType : 'std_msgs/Bool'
});

// Odometry topic
var odometry_subscriber = new ROSLIB.Topic({
	ros : ros,
	name : namespace+'/rb1_base_control/rb1_odom',
	messageType : 'nav_msgs/Odometry'
});

// Motor Status topic
var motor_status_subscriber = new ROSLIB.Topic({
	ros : ros,
	name : namespace+'/robotnik_base_hw/status',
	messageType : 'robotnik_msgs/RobotnikMotorsStatus'
});

 //gps topic
var gps_subscriber = new ROSLIB.Topic({
	ros : ros,
	name : namespace+'/fix',
	messageType : 'sensor_msgs/NavSatFix'
});



// Message Handlers
// ----------------------

// Battery
	battery_subscriber.subscribe(function(message) {
	battery_level = message.data;
	});

// IMU State
imuStateListener.subscribe(function(message) {
	
	
	//console.log(message.data);
	if ( message.data == true && !imu_state ){ 
		document.querySelector('#imu_status span').innerHTML = "<img width=30  height=30 src=images/light-green-flash.jpg border=\"0\">";
		imu_state = true;
		
	}
	if ( message.data == false && imu_state){
		document.querySelector('#imu_status span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif border=\"0\">";
		imu_state = false;
	}
});
	
// IMU temperature listener (Arduimu)
imu_temperature_subscriber.subscribe(function(message){
		imu_temperature = message.temperature;
});
	
	// Emergency Stop Listener
emergency_stop_subscriber.subscribe(function(message){
	
	if ( message.data) {
		document.querySelector('#emergency_stop span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif>";
		//emergency_button_status = true;
	}
	else if ( !message.data) {
		document.querySelector('#emergency_stop span').innerHTML = "<img width=30 height=30 src=images/light-green-flash.jpg>";
		//emergency_button_status = false;
	}
});

// Odometry Listener
odometry_subscriber.subscribe(function(message){
	
	
	x_position = message.pose.pose.position.x;
	y_position = message.pose.pose.position.y;
	z_position = message.pose.pose.position.z;

	// Get euler angles from quaternion
	var quaternion = new THREE.Quaternion(message.pose.pose.orientation.x,
		message.pose.pose.orientation.y,
		message.pose.pose.orientation.z,
		message.pose.pose.orientation.w);
        var euler = new THREE.Euler().setFromQuaternion(quaternion);
	
	x_orientation = euler.x.toFixed(4);
	y_orientation = euler.y.toFixed(4);
	z_orientation = euler.z.toFixed(4);
	
	
});

// Motor Status Listener
motor_status_subscriber.subscribe(function(message){
	flw_data[0] = message.motor_status[0].state;
	flw_data[1] = message.motor_status[0].status;
	flw_data[2] = message.motor_status[0].statusword;
	flw_data[3] = message.motor_status[0].driveflags;
	
	blw_data[0] = message.motor_status[1].state;
	blw_data[1] = message.motor_status[1].status;
	blw_data[2] = message.motor_status[1].statusword;
	blw_data[3] = message.motor_status[1].driveflags;	
	
		
	// space is required for the comparison
	if(flw_data[1] == "OPERATION_ENABLED"){
		document.querySelector('#front_left_wheel_status span').innerHTML = "<img width=30 height=30 src=images/light-green-flash.jpg>";
	}else{
		document.querySelector('#front_left_wheel_status span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif>";	
	}
	
	if(blw_data[1] == "OPERATION_ENABLED"){
		document.querySelector('#front_right_wheel_status span').innerHTML = "<img width=30 height=30 src=images/light-green-flash.jpg>";
	}else{
		document.querySelector('#front_right_wheel_status span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif>";	
	}
	
	
});

gps_subscriber.subscribe(function(message) {
	
	document.querySelector('#gps_latitude span').innerHTML =  message.latitude;
	document.querySelector('#gps_longitude span').innerHTML =  message.longitude;
	document.querySelector('#gps_altitude span').innerHTML =  message.altitude;
			
});
	

// CAMERA FRONT
function init_camera_front(){	
	  
	camera_front_pan= 0.0;
	camera_front_tilt= -(45.0)*Math.PI/180;
	camera_front_zoom = 0.0;	
	$( "#slider_camera_front_zoom" ).slider( "value", 0 );
	var pos = new ROSLIB.Message({
		pan: 0.0,
		tilt: camera_front_tilt,
		zoom: 0.0,
		relative: false
		
	});
	camera_front_ptz_publisher.publish(pos);
}

function move_camera_front_right(){
	
		if(camera_front_zoom>4000){
			camera_front_pan = -1.25*Math.PI/180;
		}else{
			camera_front_pan= -5*Math.PI/180;
		}
	camera_front_tilt = 0;
	move_camera_front();
	
}

function move_camera_front_left(){
	
		if(camera_front_zoom>4000){
			camera_front_pan = 1.25*Math.PI/180;
		}else{
			camera_front_pan= 5*Math.PI/180;
		}

	camera_front_tilt = 0;
	move_camera_front();
}

function move_camera_front_top(){
	  
		if(camera_front_zoom > 4000){
			camera_front_tilt = -1.25*Math.PI/180;
		}
		else{ 
			camera_front_tilt = -5*Math.PI/180;
		}		

	camera_front_pan = 0;
	move_camera_front();
}
		
function move_camera_front_dowm(){
	  
	  	if(camera_front_zoom > 4000){
			camera_front_tilt = 1.25*Math.PI/180;
		}
		else{ 
			camera_front_tilt = 5*Math.PI/180;
		}
    camera_front_pan = 0;
	move_camera_front();
}

		
function move_camera_front(){
	var pos = new ROSLIB.Message({
		pan: camera_front_pan,
		tilt: camera_front_tilt,
		zoom: camera_front_zoom,
		relative: true
		
	});
	camera_front_ptz_publisher.publish(pos);
	     
}
// CAMERA REAR
function init_camera_rear(){	
	  
	camera_rear_pan= 0.0;
	camera_rear_tilt= -45.0;
	camera_rear_zoom = 0.0;	
	$( "#slider_camera_front_zoom" ).slider( "value", 0 );
	var pos = new ROSLIB.Message({
		pan: 0.0,
		tilt: camera_rear_tilt,
		zoom: 0.0,
		relative: false
		
	});
	camera_rear_ptz_publisher.publish(pos);
}

function move_camera_rear_right(){
	
	if(camera_rear_pan < pan_max){
		if(camera_rear_zoom>4000){
			camera_rear_pan = camera_rear_pan + 2.5;
		}else{
			camera_rear_pan=camera_rear_pan + 10.0;
		}
	}
	move_camera_rear();
	
}

function move_camera_rear_left(){
	
	if(camera_rear_pan > pan_min){
		if(camera_rear_zoom>4000){
			camera_rear_pan = camera_rear_pan - 2.5;
		}else{
			camera_rear_pan= camera_rear_pan - 10.0;
		}
	}
	move_camera_rear();
}

function move_camera_rear_top(){
	  
	  
	//limit to 90º?
	if(camera_rear_tilt < tilt_max ){
		if(camera_rear_zoom > 4000){
			camera_rear_tilt = camera_rear_tilt + 2.5;
		}
		else{ 
			camera_rear_tilt = camera_rear_tilt + 10.0;
		}		
	}
	move_camera_rear();
}
		
function move_camera_rear_dowm(){
	  
	if ( camera_rear_tilt > tilt_min ){  
		if(camera_rear_zoom > 4000){
			camera_rear_tilt = camera_rear_tilt - 2.5;
		}
		else{ 
			camera_rear_tilt = camera_rear_tilt - 10.0;
		}
    }
	move_camera_rear();
}

		
function move_camera_rear(){
	var pos = new ROSLIB.Message({
		pan: camera_rear_pan,
		tilt: camera_rear_tilt,
		zoom: camera_rear_zoom,
		relative: false
		
	});
	camera_rear_ptz_publisher.publish(pos);
	     
}

function goIndex(){
    window.location.href = "index.html";
}

function StopRobot(){
	

	

	var aux2 = new ROSLIB.Topic({  
		ros:ros,
		name : namespace+'/cmd_vel',
		messageType : 'geometry_msgs/Twist'
	});


	var twist = new ROSLIB.Message({
	linear : {
		x : 0.0,
		y : 0.0,
		z : 0.0
		},
	angular : {
		x : 0.0,
		y : 0.0,
		z : 0.0
	}
	});

	aux2.publish(twist);  
	
		 }

	 function Forward(){
	

	

	var aux2 = new ROSLIB.Topic({  
		ros:ros,
		name : namespace+'/cmd_vel',
		messageType : 'geometry_msgs/Twist'
	});


	var twist = new ROSLIB.Message({
	linear : {
		x : 0.2,
		y : 0.0,
		z : 0.0
		},
	angular : {
		x : 0.0,
		y : 0.0,
		z : 0.0
	}
	});

	aux2.publish(twist);  
	
		 }


 function Back(){

	var aux2 = new ROSLIB.Topic({  
		ros:ros,
		name : namespace+'/cmd_vel',
		messageType : 'geometry_msgs/Twist'
	});


	var twist = new ROSLIB.Message({
	linear : {
		x : -0.2,
		y : 0.0,
		z : 0.0
		},
	angular : {
		x : 0.0,
		y : 0.0,
		z : 0.0
	}
	});

	aux2.publish(twist); 

		 }
	 function Right(){


	var aux2 = new ROSLIB.Topic({  
		ros:ros,
		name : namespace+'/cmd_vel',
		messageType : 'geometry_msgs/Twist'
	});


	var twist = new ROSLIB.Message({
	linear : {
		x : 0.0,
		y : 0.0,
		z : 0.0
		},
	angular : {
		x : 0.0,
		y : 0.0,
		z : -0.2
	}
	});

	aux2.publish(twist); 

		 }
	 function Left(){



	var aux2 = new ROSLIB.Topic({  
		ros:ros,
		name : namespace+'/cmd_vel',
		messageType : 'geometry_msgs/Twist'
	});


	var twist = new ROSLIB.Message({
	linear : {
		x : 0.0,
		y : 0.0,
		z : 0.0
		},
	angular : {
		x : 0.0,
		y : 0.0,
		z : 0.2
	}
	});

	aux2.publish(twist); 

		 }
		 
	 function MoveRobot(linear_x, angular_z){



	var aux2 = new ROSLIB.Topic({  
		ros:ros,
		name : namespace+'/cmd_vel',
		messageType : 'geometry_msgs/Twist'
	});


	var twist = new ROSLIB.Message({
	linear : {
		x : linear_x,
		y : 0.0,
		z : 0.0
		},
	angular : {
		x : 0.0,
		y : 0.0,
		z : angular_z
	}
	});

	aux2.publish(twist); 

		 }
		 function startRecording(){
	   
	   	if(is_recording == 0)
			is_recording = 1;

	   var aux2 = new ROSLIB.Service({  
		ros : ros,
		name : namespace+'/start_recording',
		messageType : 'robot_web/RecordVideo'
		});
	
		var data = new ROSLIB.ServiceRequest({
			startRecording : 1
		});
		
		aux2.callService(data,function(res){
			console.log("start recording");
			filename = res.filename;
			
			
		});
		updateStrings(filename = "");
	}
	
	function stopRecording(){
	   
	   if(is_recording == 1)
			is_recording = 0;
	   
	   var aux2 = new ROSLIB.Service({  
		ros : ros,
		name : namespace+'/start_recording',
		messageType : 'robot_web/RecordVideo'
		});
	
		var data = new ROSLIB.ServiceRequest({
			startRecording : 0
		});
		
		aux2.callService(data,function(res){
			console.log("stop recording");
			filename = res.filename;
			console.log('Saved file on ' + res.filename);
			updateStrings(filename);
		});
		
		
	}
	
	function saveImage(){
		
		//console.log("Saving Image");
		
		var aux2 = new ROSLIB.Service({  
			ros : ros,
			name : namespace+'/start_recording',
			messageType : 'robot_web/RecordVideo'
		});
		
		var data = new ROSLIB.ServiceRequest({
			startRecording : 2
		});
		
		aux2.callService(data,function(res){
			//console.log("Image saved");

			//document.querySelector('#imageStatus').innerHTML = "Image saved at " + res.filename;
		});
		
	}
	
	
/*	function callkeydownhandler(evnt, event2) {
	   var ev = (evnt) ? evnt : event;
	   var ev2 = (evnt) ? evnt : event2;
	   var code=(ev.which) ? ev.which : event.keyCode;
	   var code2=(ev.which) ? ev.which : event.keyCode;
		
	   var linear_x = 0;
	   var angular_z = 0;
	   var linear = 0.6;
           var angular = 0.6;
	   if(code == 37) // left
		angular_z += angular;
	   else if(code2 == 38)
	        linear_x += linear;
	   else if(code == 39)
		angular_z -= angular;
	   else if(code2 == 40 )
		linear_x -= linear;
		MoveRobot(linear_x, angular_z);
	}
	

	if (window.document.addEventListener) {
	   window.document.addEventListener("keydown", callkeydownhandler, false);
	} else {
	   window.document.attachEvent("onkeydown", callkeydownhandler);
	}
*/	
	
	function updateStrings(filename)
	{
		if(is_recording == 0)
			//document.querySelector('#videoStatus').innerHTML = filename + ' recorded';
		else if(is_recording == 1)			
			//document.querySelector('#videoStatus').innerHTML = 'Recording video';
	}
	
	
	function setMaximumAngle()
	{
		console.log("Set maximum angle");
		maxAngle = $("#max_angle_spinner").spinner("value");
		maxAngle = parseFloat(maxAngle);
		console.log( maxAngle );
					
		max_angle_message = new ROSLIB.Message({
		 data : maxAngle
		});
		
		//set parameter
		//max_angle_param.set(maxAngle);
	}
	
	function setTrimAngle()
	{
		console.log("Set trim angle");
		trim_angle = $("#trim_spinner").spinner("value");
		trim_angle = parseFloat(trim_angle);
								
		trim_angle_message = new ROSLIB.Message({
		 data : trim_angle
		});
		
		//set parameter
		//trim_angle_param.set(trim_angle);
	}
	
	function saveSettings()
	{
		maxAngle = $("#max_angle_spinner").spinner("value");
		maxAngle = parseFloat(maxAngle);
		trim_angle = $("#trim_spinner").spinner("value");
		trim_angle = parseFloat(trim_angle);
		min_radius = $("#min_radius_spinner").spinner("value");
		min_radius = parseFloat(min_radius);
		//reverse_direction = document.getElementById("reverse_direction_checkbox").checked;
		//console.log(maxAngle + " " + trim_angle + " " + min_radius + " " + reverse_direction);
		
		
		
	}
	
	function deactivateIMU(){
		deactivate_imu_msg.data = true;
		//document.querySelector('#imu_connected span').innerHTML = "NO";
	}
	
	function activateIMU(){
		deactivate_imu_msg.data = false;
		//document.querySelector('#imu_connected span').innerHTML = "SI";
	}
	
	function reset_odometry(){
				
		var aux2 = new ROSLIB.Service({  
			ros : ros,
			name : namespace+'/set_odometry',
			messageType : 'robotnik_msgs/set_odometry'
		});
		
		var data = new ROSLIB.ServiceRequest({
			x : 0.0,
			y : 0.0,
			z : 0.0,
			orientation : 0.0
		});
		
		aux2.callService(data,function(res){
						
		});
		
	}
	
	function saveControllerOptions(){
		
		//get options from ui
		
		//take_over = document.getElementById("take_over_checkbox").checked;
		
		//x_wam = document.getElementById("robot_type").value;
		
		//kinematic_mode = document.getElementById("kinematic_mode").value;	
		
		gearbox_reduction = $("#gearbox_reduction_spinner").spinner("value");
		gearbox_reduction = parseFloat(gearbox_reduction);
		
		diameter_wheel = $("#diameter_wheel_spinner").spinner("value");
		diameter_wheel = parseFloat(diameter_wheel);
		
		//motion_odometry = document.getElementById("motion_odometry_checkbox").checked;
		if (motion_odometry) 
			motion_odometry = "true";
		else
			motion_odometry = "false";
			
		//motors_encoders = document.getElementById("motors_encoders_checkbox").checked;
		if (motors_encoders) 
			motors_encoders = "true";
		else
			motors_encoders = "false";
			
		motors_encoders_factor = $("#motors_encoders_factor_spinner").spinner("value");
		motors_encoders_factor = parseFloat(motors_encoders_factor);
		
		
		
		var aux2 = new ROSLIB.Service({  
			ros : ros,
			name : namespace+'/set_controller_options',
			messageType : 'robot_web/set_controller_options'
		});
		
		var data = new ROSLIB.ServiceRequest({
			takeOver : take_over,
			kinematicMode: kinematic_mode,
			gearboxReduction : gearbox_reduction,
			diameterWheel : diameter_wheel,
			motionOdometry : motion_odometry,
			motorsEncoder : motors_encoders,
			motorsEncoderFactor : motors_encoders_factor,
			xWam : x_wam
		});
		
		aux2.callService(data,function(res){});
		
	}
	
	function update_controller_options(){
		var aux2 = new ROSLIB.Service({  
			ros : ros,
			name : namespace+'/get_controller_options',
			messageType : 'robot_web/get_controller_options'
		});
		var data = new ROSLIB.ServiceRequest({});
		
		aux2.callService(data,function(res){
			console.log(res.takeOver);
			console.log(res.kinematicMode);
			console.log(res.gearboxReduction);
			console.log(res.diameterWheel);
			console.log(res.motionOdometry);
			console.log(res.motorsEncoder);
			console.log(res.motorsEncoderFactor);
			console.log(res.xWam);
			
			
			if(res.motionOdometry == "true")
				motion_odometry = true;
			else
				motion_odometry = false;
				
			if(res.motorsEncoder == "true")
				motors_encoder = true;
			else
				motors_encoder = false;
			
			//document.getElementById("take_over_checkbox").checked = res.takeOver;
			
			
			//document.getElementById("motion_odometry_checkbox").checked = motion_odometry;
			
			
			//document.getElementById("motors_encoders_checkbox").checked = motors_encoder;
			
			$("#gearbox_reduction_spinner").spinner("value",res.gearboxReduction);
			$("#diameter_wheel_spinner").spinner("value",res.diameterWheel);
			$("#motors_encoders_factor_spinner").spinner("value",res.motorsEncoderFactor);
			//document.getElementById("robot_type").value = res.xWam;
			//document.getElementById("kinematic_mode").value = res.kinematicMode;
			});
		
	}
	
	function resetDriverHistory()
	{
		
		for(var i=0; i<status_codes.length; i++)
		{
			flw_data_history[2] = "0000000000000000";
			frw_data_history[2] = "0000000000000000";
			blw_data_history[2] = "0000000000000000";
			brw_data_history[2] = "0000000000000000";
		}
		for(var i=0; i<drive_status_codes.length; i++)
		{
			flw_data_history[3] = "00000000000000000000000000000000000000000000000000000000000000000";
			frw_data_history[3] = "00000000000000000000000000000000000000000000000000000000000000000";
			blw_data_history[3] = "00000000000000000000000000000000000000000000000000000000000000000";
			brw_data_history[3] = "00000000000000000000000000000000000000000000000000000000000000000";
		}
	};
	
	function mainLoop(){
		
	    
	    //min valuye = 23, max value = 27.2
	    battery_level_corrected = battery_level - minimum_battery_level;
	    $("#progressbar_battery").progressbar({value: battery_level_corrected});
	    

	    
	    // update battery
	    // -------------
	    battery_level = Math.round(battery_level * 100) / 100;
		//document.querySelector('#battery_status span').innerHTML = battery_level;
	
		if(battery_level < 23.0 && battery_status ) {
			//document.querySelector('#battery_ok span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif>";
			battery_status = false;
			//console.log("Bateria not ok");
		}
		else if(battery_level >= 23.0 && !battery_status ) {
			//document.querySelector('#battery_ok span').innerHTML = "<img width=30 height=30 src=images/light-green-flash.jpg>";
			//console.log("Bateria ok");
			battery_status = true;
		}
	    
	    
	    // update imu temperature
	    // -------------
		imu_temperature = Math.round(imu_temperature * 100) / 100;
		//document.querySelector('#imu_temperature span').innerHTML = imu_temperature;
		
		if ( imu_temperature < limit_temperature && !imu_temperature_status) {
			//document.querySelector('#imu_temperature_alarm span').innerHTML = "<img width=30 height=30 src=images/light-green-flash.jpg>";
			imu_temperature_status = true;
		}
		else if (imu_temperature >= limit_temperature && imu_temperature_status)
			//document.querySelector('#imu_temperature_alarm span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif>";
			imu_temperature_status = false;
	    
	    		    
		// update odometry
	    // -------------
	    
		x_position = Math.round(x_position * 10000) / 10000;
		y_position = Math.round(y_position * 10000) / 10000;
		z_position = Math.round(z_position * 10000) / 10000;
		
		x_orientation = Math.round(x_orientation * 10000) / 10000;
		y_orientation = Math.round(y_orientation * 10000) / 10000;
		z_orientation = Math.round(z_orientation * 10000) / 10000;
		
		if (x_position == 0) x_position = x_position.toFixed(4);
		if (y_position == 0) y_position = y_position.toFixed(4);
		if (z_position == 0) z_position = z_position.toFixed(4);
		if (x_orientation == 0) x_orientation = x_orientation.toFixed(4);
		if (y_orientation == 0) y_orientation = y_orientation.toFixed(4);
		if (z_orientation == 0) z_orientation = z_orientation.toFixed(4);
		
		
		/*
                document.querySelector('#odometry_position_x span').innerHTML = x_position;
		document.querySelector('#odometry_position_y span').innerHTML = y_position;
		document.querySelector('#odometry_position_z span').innerHTML = z_position;
		
		document.querySelector('#odometry_orientation_x span').innerHTML = x_orientation;
		document.querySelector('#odometry_orientation_y span').innerHTML = y_orientation;
		document.querySelector('#odometry_orientation_z span').innerHTML = z_orientation;
	    */
	    //update motor values
	    
	    // Left Motor
		// ----------------
	    //document.querySelector('#flw_state span').innerHTML = flw_data[0];
		//document.querySelector('#flw_status span').innerHTML = flw_data[1];
		
		status_word_string = "";
		status_word_sub_string = "";
		for( var i = 0; i < flw_data[2].length ; i++)
		{
				if(flw_data[2][i] == 1)
				{
					status_word_sub_string = status_codes[i];
					status_word_string = status_word_string.concat("<br><b>",status_word_sub_string,"</b>");
					
					// updating history
					flw_data_history[2] = flw_data_history[2].substr(0, i) + '1' + flw_data_history[2].substr(i + 1);
				}
				else 
				{
					if(flw_data_history[2][i] == 1) // flag used in the past
					{
						status_word_sub_string = status_codes[i];
						status_word_string = status_word_string.concat("<br><ins>",status_word_sub_string,"</ins>");
						//console.log("flag used in the past");
					}
					else // flag off
					{
						status_word_sub_string = status_codes[i];
						status_word_string = status_word_string.concat("<br>",status_word_sub_string);
					}
				}
		};
		//document.querySelector('#flw_status_words span').innerHTML = status_word_string;

		status_word_string = "";
		status_word_sub_string = "";
		for( var i = 0; i < flw_data[3].length ; i++)
		{
				if(flw_data[3][i] == 1)
				{
					status_word_sub_string = drive_status_codes[i];
					status_word_string = status_word_string.concat("<br><b>",status_word_sub_string,"</b>");
				
					// updating history
					flw_data_history[3] = flw_data_history[3].substr(0, i) + '1' + flw_data_history[3].substr(i + 1);
				}
				else 
				{
					if(flw_data_history[3][i] == 1) // flag used in the past
					{
						status_word_sub_string = drive_status_codes[i];
						status_word_string = status_word_string.concat("<br><ins>",status_word_sub_string,"</ins>");
						//console.log("flag used in the past");
					}
					else // flag off
					{
						status_word_sub_string = drive_status_codes[i];
						status_word_string = status_word_string.concat("<br>",status_word_sub_string);
					}
				}
		};
		//document.querySelector('#flw_driver_status_words span').innerHTML = status_word_string;
		
		
		// Right Motor
		// ----------------
		
		//document.querySelector('#frw_state span').innerHTML = blw_data[0];
		//document.querySelector('#frw_status span').innerHTML = blw_data[1];
		
		// status codes
		// -------------
		status_word_string = "";
		status_word_sub_string = "";
						
		for( var i = 0; i < blw_data[2].length ; i++)
		{
				if(blw_data[2][i] == 1)
				{
					status_word_sub_string = status_codes[i];
					status_word_string = status_word_string.concat("<br><b>",status_word_sub_string,"</b>");
					
					// updating history
					blw_data_history[2] = blw_data_history[2].substr(0, i) + '1' + blw_data_history[2].substr(i + 1);
				}
				else 
				{
					if(blw_data_history[2][i] == 1) // flag used in the past
					{
						status_word_sub_string = status_codes[i];
						status_word_string = status_word_string.concat("<br><ins>",status_word_sub_string,"</ins>");
						//console.log("flag used in the past");
					}
					else // flag off
					{
						status_word_sub_string = status_codes[i];
						status_word_string = status_word_string.concat("<br>",status_word_sub_string);
					}
				}
		};
		//document.querySelector('#frw_status_words span').innerHTML = status_word_string;
		status_word_string = "";
		status_word_sub_string = "";
		for( var i = 0; i < blw_data[3].length ; i++)
		{
				if(blw_data[3][i] == 1)
				{
					status_word_sub_string = drive_status_codes[i];
					status_word_string = status_word_string.concat("<br><b>",status_word_sub_string,"</b>");
				
					// updating history
					blw_data_history[3] = blw_data_history[3].substr(0, i) + '1' + blw_data_history[3].substr(i + 1);
				}
				else 
				{
					if(blw_data_history[3][i] == 1) // flag used in the past
					{
						status_word_sub_string = drive_status_codes[i];
						status_word_string = status_word_string.concat("<br><ins>",status_word_sub_string,"</ins>");
						//console.log("flag used in the past");
					}
					else // flag off
					{
						status_word_sub_string = drive_status_codes[i];
						status_word_string = status_word_string.concat("<br>",status_word_sub_string);
					}
				}
		};
		//document.querySelector('#frw_driver_status_words span').innerHTML = status_word_string;
	    
	} 
    
    monitorListener.subscribe(function(message){
		clearSystemMonitorData();
	    
	    createCpuTemperature(message);
            			
		createCpuUsageTable(message);
            
		createMemUsageTable(message);
           
		createNetUsageTable(message);
           
		createHddUsageTable(message);    
        });
	
	function clearSystemMonitorData(){/*
		var myNode = document.getElementById("cpu_usage_tb");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }
	    myNode = document.getElementById("cpu_temperature");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }
	    myNode = document.getElementById("net_usage_tb");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }
	    myNode = document.getElementById("hdd_usage_tb");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }
	    myNode = document.getElementById("mem_usage_tb");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }
	    myNode = document.getElementById("cpu_usage_ld1");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }
	    myNode = document.getElementById("mem_total");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }
	    myNode = document.getElementById("mem_free");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }
	    myNode = document.getElementById("mem_used");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }
	    myNode = document.getElementById("cpu_usage_ld5");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }
	    myNode = document.getElementById("cpu_usage_ld15");
	    while (myNode.firstChild) {
			myNode.removeChild(myNode.firstChild);
	    }*/
	}
	
	function createCpuTemperature(message){
		// Creating CPU Temperature fields
		var num_cores = message.diagCpuTemp.status.cores.length;
		for(var i =0;i< num_cores ;i++){
			var ele = document.createElement('div');
			var id = 'temperature_core_'+(i+1);
			ele.setAttribute('id', id);
			//ele.innerHTML = 'Temperature (Core '+(i+1) + ') = ';
			var span = document.createElement('span');
			//span.innerHTML = message.diagCpuTemp.status.cores[i].temp;
			//document.getElementById('cpu_temperature').appendChild(ele);
			//document.getElementById('temperature_core_'+(i+1)).appendChild(span);
		}	
	}
	 
	function createMemUsageTable(message){
		// Creating Memory table
        var tblBody = document.createElement('tbody');
        var num_memories = message.diagMem.status.memories.length;
        var keys = Object.keys(message.diagMem.status.memories[num_memories-1]);
        keys.splice(keys.indexOf('name'),1);
        // The first row contains the memories name
        var row = document.createElement('tr');
        var cell = document.createElement('td');
        cell.setAttribute("style","width: 6em");
        row.appendChild(cell);
        for(var i=0;i<num_memories;i++){
			cell = document.createElement('td');
			cell.setAttribute("style","width: 6em");
			var cellText = document.createTextNode(message.diagMem.status.memories[i]['name']);
			cell.appendChild(cellText);
			row.appendChild(cell);
		}
		tblBody.appendChild(row);
		// Creating the others rows
		for(var i=0;i<keys.length;i++){
			row = document.createElement('tr');
			cell = document.createElement('td');
			var cellText = document.createTextNode(keys[i]);
			cell.appendChild(cellText);
			row.appendChild(cell);
			for(var j = 0;j<num_memories;j++){
				cell = document.createElement('td');
				cell.setAttribute('id', 'mem_usage_'+j+'_'+keys[i]);
				cellText = document.createTextNode(message.diagMem.status.memories[j][keys[i]]);
				cell.appendChild(cellText);
				row.appendChild(cell);
			}
			tblBody.appendChild(row);
		}
		/*document.getElementById('mem_usage_tb').appendChild(tblBody);
		
		var cellText = document.createTextNode("Total = " + message.diagMem.status.totalM);
		document.getElementById('mem_total').appendChild(cellText);
		cellText = document.createTextNode("Used = " + message.diagMem.status.usedM);
		document.getElementById('mem_used').appendChild(cellText);
		cellText = document.createTextNode("Free = " + message.diagMem.status.freeM);
		document.getElementById('mem_free').appendChild(cellText);	*/
	  }
	
	function createCpuUsageTable(message){
		// Creating CPU Usage table
		var tblBody = document.createElement('tbody');
		var num_cores = message.diagCpuUsage.status.cores.length;
		var keys = Object.keys(message.diagCpuUsage.status.cores[num_cores-1]);
		keys.splice(keys.indexOf('id'),1);
		// The first row contains the cores id
		var row = document.createElement('tr');
		var cell = document.createElement('td');
		cell.setAttribute("style","width: 6em");
		row.appendChild(cell);
		for(var i=0;i<num_cores;i++){
			cell = document.createElement('td');
			cell.setAttribute("style","width: 6em");
			var cellText = document.createTextNode("Core "+i);
			cell.appendChild(cellText);
			row.appendChild(cell);
		}
		tblBody.appendChild(row);
		// Creating the others rows
		for(var i=0;i<keys.length;i++){
			row = document.createElement('tr');
			cell = document.createElement('td');
			var cellText = document.createTextNode(keys[i]);
			cell.appendChild(cellText);
			row.appendChild(cell);
			for(var j = 0;j<num_cores;j++){
				cell = document.createElement('td');
				cell.setAttribute('id', 'cpu_usage_'+j+'_'+keys[i]);
				if (typeof message.diagCpuUsage.status.cores[j][keys[i]] === 'number'){
					 cellText = document.createTextNode(Math.round(message.diagCpuUsage.status.cores[j][keys[i]] * 100)/100);
				}else{
					 cellText = document.createTextNode(message.diagCpuUsage.status.cores[j][keys[i]]);
				}
				cell.appendChild(cellText);
				row.appendChild(cell);
			}
			tblBody.appendChild(row);
		}
		//document.getElementById('cpu_usage_tb').appendChild(tblBody);
		
		var cellText = document.createTextNode("load_avg1 = " + message.diagCpuUsage.status.load_avg1);
		//document.getElementById('cpu_usage_ld1').appendChild(cellText);
		cellText = document.createTextNode("load_avg5 = " + message.diagCpuUsage.status.load_avg5);
		//document.getElementById('cpu_usage_ld5').appendChild(cellText);
		cellText = document.createTextNode("load_avg15 = " + message.diagCpuUsage.status.load_avg15);
		//document.getElementById('cpu_usage_ld15').appendChild(cellText);	
	}
	
	function createNetUsageTable(message){
		// Creating Net table
		var tblBody = document.createElement('tbody');
		var num_interfaces = message.diagNet.status.interfaces.length;
		var keys = Object.keys(message.diagNet.status.interfaces[num_interfaces-1]);
		keys.splice(keys.indexOf('name'),1);
		// The first row contains the interfaces name
		var row = document.createElement('tr');
		var cell = document.createElement('td');
		cell.setAttribute("style","width: 6em");
		row.appendChild(cell);
		for(var i=0;i<num_interfaces;i++){
			cell = document.createElement('td');
			cell.setAttribute("style","width: 6em");
			var cellText = document.createTextNode(message.diagNet.status.interfaces[i]['name']);
			cell.appendChild(cellText);
			row.appendChild(cell);
		}
		tblBody.appendChild(row);
		// Creating the others rows
		for(var i=0;i<keys.length;i++){
			row = document.createElement('tr');
			cell = document.createElement('td');
			var cellText = document.createTextNode(keys[i]);
			cell.appendChild(cellText);
			row.appendChild(cell);
			for(var j = 0;j<num_interfaces;j++){
				cell = document.createElement('td');
				cell.setAttribute('id', 'net_usage_'+j+'_'+keys[i]);
				if (typeof message.diagNet.status.interfaces[j][keys[i]] === 'number'){
					 cellText = document.createTextNode(Math.round(message.diagNet.status.interfaces[j][keys[i]] * 100)/100);
				}else{
					 cellText = document.createTextNode(message.diagNet.status.interfaces[j][keys[i]]);
				}
				cell.appendChild(cellText);
				row.appendChild(cell);
			}
			tblBody.appendChild(row);
		}
		//document.getElementById('net_usage_tb').appendChild(tblBody);	
	}
		
	function createHddUsageTable(message){
		// Creating HDD table
		var tblBody = document.createElement('tbody');
		var num_disks = message.diagHdd.status.disks.length;
		var keys = Object.keys(message.diagHdd.status.disks[num_disks-1]);
		keys.splice(keys.indexOf('name'),1);
		// The first row contains the disks name
		var row = document.createElement('tr');
		var cell = document.createElement('td');
		cell.setAttribute("style","width: 6em");
		row.appendChild(cell);
		for(var i=0;i<num_disks;i++){
			cell = document.createElement('td');
			cell.setAttribute("style","width: 6em");
			cellText = document.createTextNode(message.diagHdd.status.disks[i]['name']);
			cell.appendChild(cellText);
			row.appendChild(cell);
		}
		tblBody.appendChild(row);
		// Creating the others rows
		for(var i=0;i<keys.length;i++){
			row = document.createElement('tr');
			cell = document.createElement('td');
			var cellText = document.createTextNode(keys[i]);
			cell.appendChild(cellText);
			row.appendChild(cell);
			for(var j = 0;j<num_disks;j++){
				cell = document.createElement('td');
				cell.setAttribute('id', 'hdd_usage_'+j+'_'+keys[i]);
				if (typeof message.diagHdd.status.disks[j][keys[i]] === 'number'){
					 cellText = document.createTextNode(Math.round(message.diagHdd.status.disks[j][keys[i]] * 100)/100);
				}else{
					 cellText = document.createTextNode(message.diagHdd.status.disks[j][keys[i]]);
				}
				cell.appendChild(cellText);
				row.appendChild(cell);
			}
			tblBody.appendChild(row);
		}
		//document.getElementById('hdd_usage_tb').appendChild(tblBody);	
    }

	//jquery init
	$(document).ready(function() {			

		if (has_camera == false){
		//	document.getElementById("CAMERA").href.value="#";
		//	document.getElementById("CAMERA").hide();
			$('#cam_title').remove();
			$('#CAMERA').remove();
			//document.getElementById("CAMERA").style.visibility = 'hidden';
			//document.getElementById("CAMERA").style.
			//document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Cam_disabled.";
        	//document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
		}

		// update displays (controller)
		update_controller_options();
		
		var yaw_angle_left_100 = 0.0;
		var yaw_angle_right_100 = 0.0;
		
		//Progress bars
		$("#progressbar_battery").progressbar({value: battery_level});
		$("#progressbar_battery" ).progressbar( "option", "max", maximum_battery_level - minimum_battery_level );
		
		$("#progressbar_left").progressbar({value: yaw_angle_left});
		$("#progressbar_right").progressbar({value: yaw_angle_right});

												
		//Spinners (elliot options
		/*
		$("#max_angle_spinner").spinner({step: 0.1, min: 0.0, max: 90.0});
		$("#max_angle_spinner").spinner("value",max_angle);
		$("#trim_spinner").spinner({step: 0.1, min: -1.0, max: 1.0});
		$("#trim_spinner").spinner("value",trim_angle);
		$("#min_radius_spinner").spinner({step: 0.1, min: 0.0, max: 10.0});
		$("#min_radius_spinner").spinner("value",1.0);
		*/
		
		//Spinners (controller options)
		$("#gearbox_reduction_spinner").spinner({step: 0.01, min: 0.0, max: 100.0});
		$("#gearbox_reduction_spinner").spinner("value",gearbox_reduction);
		$("#diameter_wheel_spinner").spinner({step: 0.001, min: 0.0, max: 1.0});
		$("#diameter_wheel_spinner").spinner("value",diameter_wheel);
		$("#motors_encoders_factor_spinner").spinner({step: 1, min: 0, max: 100000});
		$("#motors_encoders_factor_spinner").spinner("value",motors_encoders_factor);
	
	
		//document.querySelector('#imu_connected span').innerHTML = "SI";
		
		
		// Sliders
		$( "#slider_camera_front_zoom" ).slider();
		$( "#slider_camera_rear_zoom" ).slider();
		$( "#slider_focus" ).slider();
		$( "#slider_camera_front_zoom" ).on( "slidechange", function( event, ui ) {
			
			
			var slider_value = $( "#slider_camera_front_zoom" ).slider( "value" );
			camera_front_zoom = slider_value * 172;
			//console.log(camera_front_zoom);
			move_camera_front();
		} );
		$( "#slider_camera_rear_zoom" ).on( "slidechange", function( event, ui ) {
			
			
			var slider_value = $( "#slider_camera_rear_zoom" ).slider( "value" );
			camera_rear_zoom = slider_value * 172;
			//console.log(camera_front_zoom);
			move_camera_rear();
		} );
			
		// init alarms
                /*
		document.querySelector('#imu_status span').innerHTML = "<img width=30 height=30 src=images/light-green-flash.jpg border=\"0\">";
		document.querySelector('#battery_ok span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif>";
		document.querySelector('#imu_temperature_alarm span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif>";
		document.querySelector('#emergency_stop span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif>";
		document.querySelector('#front_left_wheel_status span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif>";
		document.querySelector('#front_right_wheel_status span').innerHTML = "<img width=30 height=30 src=images/light-red-flash.gif>";
		*/
		//init messages
		max_angle_message = new ROSLIB.Message({
		 data : max_angle
		});
		
		trim_angle_message = new ROSLIB.Message({
		 data : trim_angle
		});
		
		deactivate_imu_msg = new ROSLIB.Message({
		 data : false
		});
		
		// Motor Dialogs
		// --------------
		
		// Front Left Motor
		// ----------------
		
		$("#dialog_flw").dialog({
			autoOpen: false
		});
		
		$("#button_flw").click(function(){
			
			//close other dialogs
			$("#dialog_flw").dialog("close");
			$("#dialog_frw").dialog("close");

			

			$("#dialog_flw").dialog("open");
		});
				
		// Front Right Motor
		// ----------------
		
		$("#dialog_frw").dialog({
			autoOpen: false
		});
		
		$("#button_frw").click(function(){
			
			//close other dialogs
			$("#dialog_flw").dialog("close");
			$("#dialog_frw").dialog("close");

			
			$("#dialog_frw").dialog("open");
		});
		
		
		
		
		// reset drivers data history
		resetDriverHistory();
		
		
		//call to mainLoop each x ms
        setInterval(mainLoop, 1000);
        

        if(!has_camera){
            $('#exTab1 ul li:last').hide();
        }
	});
