<div class="backgroudPagWeb" style="margin-top:60px">
    <div class="row">
        <div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
        <div class="row">
                <div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
                    
                    <center>
                        <p><label>Loock joystick controls</label></p>
                        <label class="switch">
                            <input type="checkbox" id = "cbActiveJoyStick">
                            <span class="slider2 round"></span>
                        </label>
                    </center>
                </div>
            </div>
            <div class="row">
                <div class=" col-xl-3 col-md-3 col-ls-3 col-xs-3">
                    <center>
                    <div id="divJoystick" class="panel-body" >
                        <img src="image/JoyStick_up_down.png"  id= "img_joystick_linear_control" class="img-responsive rounded-circle center" style="width:100%"/>
                    </div>
                    </center>
                </div>
                <div class=" col-xl-6 col-md-6 col-ls-6 col-xs-6">
                    <center>
                        <div class="row">
                            <div class=" col-xl-12 col-md-12 col-ls-12 col-xs-12">
                                <div class="text-center">
                                    <img id="camera_video" width="100%" height="100%" src="http://<?php 
				    if(substr(gethostbyname($_SERVER['SERVER_NAME']),0,7)=="192.168"){ 
					echo "192.168.0.185/axis-cgi/mjpg/video.cgi?streamprofile=viesgo"; 
				    }else{
					echo gethostbyname ($_SERVER['SERVER_NAME']).":2000/axis-cgi/mjpg/video.cgi?streamprofile=viesgo4g";
				    }
				    ?>">
                                    <!--img id="camera_video" width="100%" height="100%" src="http://192.168.0.185/axis-cgi/mjpg/video.cgi?resolution=320x240&videocodec=h264&amp;streamprofile=Bandwidth"-->
                                    <!--video id="video" src="rtsp://192.168.0.185/axis-media/media.amp?videocodec=h264" autoplay="autoplay" width="videowidth" height="videoheight"></video-->
                                </div>
                        </div>
                        <div class="row">
                            <hr>
                            <div class=" col-xl-4 col-md-4 col-ls-4 col-xs-4">
                                <button type="button" class="btn btn-default btn-sm" onclick="zoomIn()">
                                    <span class="glyphicon glyphicon-zoom-in"></span>Zoom In 
                                </button>
                            </div>
                            <div class=" col-xl-4 col-md-4 col-ls-4 col-xs-4">
                                <button type="button" class="btn btn-default btn-sm" onclick="centerCamera()">
                                    <span class="glyphicon glyphicon-screenshot"></span> Center cam 
                                </button>
                            </div>
                            <div class=" col-xl-4 col-md-4 col-ls-4 col-xs-4">
                                <button type="button" class="btn btn-default btn-sm" onclick="zoomOut()">
                                    <span class="glyphicon glyphicon-zoom-out"></span>Zoom Out 
                                </button>
                            </div>
                        </div>

                    </center>
                </div>
                <div class=" col-xl-3 col-md-3 col-ls-3 col-xs-3">
                    <center>
                        <div id="divJoystick" class="panel-body" >
                            <img src="image/JoyStick_left_right.png"  id= "img_joystick_angular_control" class="img-responsive rounded-circle center" style="width:100%"/>
                        </div>
                    </center>
                </div>
            </div>
            
        </div>
    </div>
</div>

<script>

var joystick_linear_control;
var joystick_angular_control;
var joystick_camera_control;
$(document).ready(function () {
    joystick_linear = true;
    joystick_linear_control = new VirtualJoystick({
        container: document.getElementById('img_joystick_linear_control'),
        mouseSupport: true,
    });

    setInterval(function () {
        if (joystick_linear_control.deltaY() != 0 && joystick_linear_control.deltaX() != 0 || !joystick_linear) {
            var twist = getStructTwist();
            var x = (-joystick_linear_control.deltaY()) / 100;
            if (x > 0.5) {
                x = 0.5;
            } else if (x < -0.5) {
                x = -0.5;
            }
            var y = (-joystick_linear_control.deltaX()) / 100;
            if (y > 0.5) {
                y = 0.5;
            } else if (y < -0.5) {
                y = -0.5;
            }
            twist.linear.x = x;
            
            if(document.getElementById('cbActiveJoyStick').checked){
                publish('<?php echo $robot_dictionary["topic_joystick_robot_control"] ?>', 'geometry_msgs/Twist', twist);
            }
            if (joystick_linear_control.deltaY() == 0 && joystick_linear_control.deltaX() == 0) {
                joystick_linear = true;
            } else {
                joystick_linear = false;
            }
        }
        ;

    }, 1 / 30 * 1000);
    
    
    joystick_angular = true;
    joystick_angular_control = new VirtualJoystick({
        container: document.getElementById('img_joystick_angular_control'),
        mouseSupport: true,
    });

    setInterval(function () {
        if (joystick_angular_control.deltaY() != 0 && joystick_angular_control.deltaX() != 0 || !joystick_angular) {
            var twist = getStructTwist();
            var x = (-joystick_angular_control.deltaY()) / 100;
            if (x > 0.5) {
                x = 0.5;
            } else if (x < -0.5) {
                x = -0.5;
            }
            var y = (-joystick_angular_control.deltaX()) / 100;
            if (y > 0.5) {
                y = 0.5;
            } else if (y < -0.5) {
                y = -0.5;
            }
            if (x != 0 && y != 0)
                twist.angular.z = (Math.atan2(y, x));
            if(document.getElementById('cbActiveJoyStick').checked){
                publish('<?php echo $robot_dictionary["topic_joystick_robot_control"] ?>', 'geometry_msgs/Twist', twist);
            }
            if (joystick_angular_control.deltaY() == 0 && joystick_angular_control.deltaX() == 0) {
                joystick_angular = true;
            } else {
                joystick_angular = false;
            }
        }
        ;

    }, 1 / 30 * 1000);


    joystick_camera = true;
    joystick_camera_control = new VirtualJoystick({
        container: document.getElementById('camera_video'),
        mouseSupport: true,
    });

   

    setInterval(function () {
        if (joystick_camera_control.deltaY() != 0 && joystick_camera_control.deltaX() != 0 || !joystick_camera) {
            var twist_camera = {
                pan:0.0,
                tilt:0.0,
                zoom:0.0,
                relative:true
            };
            var max_inc = 0.1;
            var x = (-joystick_camera_control.deltaY()) / 1000;
            if (x > max_inc) {
                x = max_inc;
            } else if (x < -max_inc) {
                x = -max_inc;
            }
            var y = (-joystick_camera_control.deltaX()) / 1000;
            if (y > max_inc) {
                y = max_inc;
            } else if (y < -max_inc) {
                y = -max_inc;
            }
            twist_camera.pan = y;
            twist_camera.tilt = x;
            publish('<?php echo $robot_dictionary["topic_joystick_axi_control"] ?>', 'robotnik_msgs/ptz', twist_camera);
            
            if (joystick_camera_control.deltaY() == 0 && joystick_camera_control.deltaX() == 0) {
                joystick_camera = true;
            } else {
                joystick_camera = false;
            }
        }
        ;

    }, 1 / 30 * 1000);

    
    
            

});

function centerCamera(){
    var twist_camera = {
        pan:0.0,
        tilt:-0.005,
        zoom:0.0,
        relative:false
    };  
    console.log("Hello world!");  
    for(cont=0; cont < 10; cont++){
        publish('<?php echo $robot_dictionary["topic_joystick_axi_control"] ?>', 'robotnik_msgs/ptz', twist_camera);
    }
}


function zoomIn(){
    zoom_value =100;
    
    var twist_camera = {
        pan:0.0,
        tilt:0.0,
        zoom:zoom_value,
        relative:true
    };  
    
    publish('<?php echo $robot_dictionary["topic_joystick_axi_control"] ?>', 'robotnik_msgs/ptz', twist_camera);
 
}

function zoomOut(){
    console.log(zoom_value);
    zoom_value =  -100;
    
    
    var twist_camera = {
        pan:0.0,
        tilt:0.0,
        zoom:zoom_value,
        relative:true
    };  
        publish('<?php echo $robot_dictionary["topic_joystick_axi_control"] ?>', 'robotnik_msgs/ptz', twist_camera);
    
}
    
</script>
