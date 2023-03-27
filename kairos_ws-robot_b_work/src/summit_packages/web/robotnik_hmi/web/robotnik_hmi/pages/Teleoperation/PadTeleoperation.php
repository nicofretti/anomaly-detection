<div class="backgroundPageWeb" style="margin-top:60px">
    <div class="container-fluid">
    
    <div class="row">
        <div class=" col-xl-12 col-md-12 col-ls-12 col-xs-12">
        
            <center>
                <div class="row">
                    <div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
                        <p><label><?php echo $lang['TELEOPERATION_EXPLANATION'];?><label></p>
                        <div class="text-center">
                            <!--img id="camera_video" class="img-responsive center-block" width="70%" height="70%" 
                          src="http://<?php echo ($_SERVER['SERVER_NAME']); ?>:<?php echo ($robot_dictionary["web_video_server_port"])?>/stream?topic=<?php echo ($robot_dictionary["topic_front_camera"])?>"/!-->
				<video src="blob:http://192.168.0.185/64496e70-3dd4-42e2-9d5f-29ab012f38bd" class="video__media" style=""></video>
                        </div>
                    </div>
                    <div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
                        <button id="fullScreen" onclick="videoFullScreen()"><?php echo $lang['TELEOPERATION_FULL_SCREEN'];?> </button>
                        <script>
                            function videoFullScreen(){
                                var featuredimg = document.getElementById("camera_video")
                                if (featuredimg.requestFullscreen){
                                    featuredimg.requestFullscreen() 
                                }
                            }
                        </script>
                    </div>
                </div>
                <div class="row">
                    <hr>
                        <div class=" col-xl-4 col-md-4 col-ls-4 col-xs-4">
                            <button type="button" class="btn btn-default btn-sm" onclick="zoomIn()">
                                <span class="glyphicon glyphicon-zoom-in"></span><?php echo $lang['TELEOPERATION_ZOOM_IN'];?> 
                            </button>
                        </div>
                        <div class=" col-xl-4 col-md-4 col-ls-4 col-xs-4">
                            <button type="button" class="btn btn-default btn-sm" onclick="centerCamera()">
                                <span class="glyphicon glyphicon-screenshot"></span><?php echo $lang['TELEOPERATION_CENTER_THE_CAMERA'];?>
                            </button>
                        </div>
                        <div class=" col-xl-4 col-md-4 col-ls-4 col-xs-4">
                            <button type="button" class="btn btn-default btn-sm" onclick="zoomOut()">
                                <span class="glyphicon glyphicon-zoom-out"></span><?php echo $lang['TELEOPERATION_ZOOM_OUT'];?>
                            </button>
                        </div>
                </div>
            </center>
        </div>
    </div>
    <div id="gamepadPrompt"></div>
    <div id="gamepadDisplay"></div>
 
    <script>
    var hasGP = false;
    var repGP;
    var ptz_pantilt_inc = 0.02;
    var ptz_zoom_inc = 100;
    var ptz_inc = 0.01;
    var ptz_pantilt_max_inc = 0.2;
    var ptz_pantilt_min_inc = 0.01;
    var show_buttons_log = true;
 
    function canGame() {
        return "getGamepads" in navigator;
    }
 
    function reportOnGamepad() {
        var gp = navigator.getGamepads()[0];
        var deadman_button = 5;
        var config_button = 7;
        var linear_speed_axis = 1;
        var angular_speed_axis = 2;
        var ptz_home_button = 9;
        var ptz_tilt_up_button = 12;
        var ptz_tilt_down_button = 13;
        var ptz_pan_right_button = 15;
        var ptz_pan_left_button = 14;
        var ptz_zoom_in_button = 6;
        var ptz_zoom_out_button = 4;
        var ptz_config_inc_button = 12;
        var ptz_config_dec_button = 13;
	
        if(gp==null){
            return;
        }
        var html = "";
            html += "id: "+gp.id+"<br/>";
 
        for(var i=0;i<gp.buttons.length;i++) {
            html+= "Button "+(i)+": ";
            if(gp.buttons[i].pressed) html+= " pressed";
            html+= "<br/>";
        }
 
        for(var i=0;i<gp.axes.length; i+=1) {
            //html+= "Stick "+(Math.ceil(i/2)+1)+": "+gp.axes[i]+","+gp.axes[i+1]+"<br/>";
            html+= "Axis "+ i +": "+gp.axes[i] +"<br/>";
        }
        //console.log(gp.buttons[deadman_button]);
        if(gp.buttons[deadman_button].pressed){
	    if(gp.buttons[config_button].pressed){
		if(gp.buttons[ptz_config_inc_button].pressed){
		    ptz_pantilt_inc += ptz_inc;
		    if(ptz_pantilt_inc > ptz_pantilt_max_inc){
			ptz_pantilt_inc = ptz_pantilt_max_inc;
		    }
		    //console.log("ptz_pantilt_inc = %f", ptz_pantilt_inc);

		}
		else if(gp.buttons[ptz_config_dec_button].pressed){
		    ptz_pantilt_inc -= ptz_inc;
		    if(ptz_pantilt_inc < ptz_pantilt_min_inc){
			ptz_pantilt_inc = ptz_pantilt_min_inc;
		    }
		}
	    }else{
            joystick(gp.axes[linear_speed_axis], gp.axes[angular_speed_axis]);
		if(gp.buttons[ptz_home_button].pressed){
		    centerCamera();
		    //console.log("homing camera!");
		}else{
		    var twist_camera = {
			pan:0.0,
			tilt:0.0,
			zoom: 0.0,
			relative:true
		    };
		    console.log(gp.buttons);
		    if(gp.buttons[ptz_tilt_up_button].pressed){
			twist_camera.tilt = -ptz_pantilt_inc;
		    }
		    if(gp.buttons[ptz_tilt_down_button].pressed){
			twist_camera.tilt = ptz_pantilt_inc;
		    }
		    if(gp.buttons[ptz_pan_right_button].pressed){
			twist_camera.pan = -ptz_pantilt_inc;
		    }
		    if(gp.buttons[ptz_pan_left_button].pressed){
			twist_camera.pan = ptz_pantilt_inc;
		    }
		    if(gp.buttons[ptz_zoom_in_button].pressed){
			twist_camera.zoom = ptz_zoom_inc;
		    }
		    if(gp.buttons[ptz_zoom_out_button].pressed){
			twist_camera.zoom = -ptz_zoom_inc;
		    }
		    if(twist_camera.zoom != 0.0 || twist_camera.tilt != 0.0 || twist_camera.pan != 0.0){
			//console.log("Setting ptz: %f, %f, %f", twist_camera.pan, twist_camera.tilt, twist_camera.zoom);
			publish('<?php echo $robot_dictionary["topic_joystick_axi_control"] ?>', 'robotnik_msgs/ptz', twist_camera);
		    }
		    
		}
	    }
	    
        }
        
	if(show_buttons_log){
	    $("#gamepadDisplay").html(html);
	}
    }
    
    var first_pad_connection = false;
 
    $(document).ready(function() {
 
        if(canGame()) {
 
            var prompt = "To begin using your gamepad, connect it and press any button!";
            $("#gamepadPrompt").text(prompt);
 
            $(window).on("gamepadconnected", function() {
                hasGP = true;
                $("#gamepadPrompt").html("Gamepad connected!");
                //console.log("connection event");
                repGP = window.setInterval(reportOnGamepad,100);
                if(first_pad_connection){
                    alertify.success('<?php echo $lang['TELEOPERATION_PAD_CONNECTED'];?>');
                }
                first_pad_connection = true;
            });
 
            $(window).on("gamepaddisconnected", function() {
                //console.log("disconnection event");
                $("#gamepadPrompt").text(prompt);
                window.clearInterval(repGP);
                alertify.error('<?php echo $lang['TELEOPERATION_PAD_DISCONNECTED'];?>');
            });
 
            //setup an interval for Chrome
            var checkGP = window.setInterval(function() {
                //console.log('checkGP');
                if(navigator.getGamepads()[0]) {
                    if(!hasGP) $(window).trigger("gamepadconnected");
                    window.clearInterval(checkGP);
                }
            }, 1/10*1000);
        }
 
    });


    function joystick (linear, angular){
        
        var twist = getStructTwist();
        twist.linear.x = linear*-0.5;
        twist.angular.z = angular*-0.5;
        publish('<?php echo $robot_dictionary["topic_joystick_robot_control"] ?>', 'geometry_msgs/Twist', twist);
	//console.log(twist.linear)
    }
    
    function centerCamera(){
        var twist_camera = {
            pan:0.0,
            tilt:-0.00,
            zoom:0.0,
            relative:false
        };  
        //console.log("Juanma saluda ;\)");  
        //for(cont=0; cont < 10; cont++){
        publish('<?php echo $robot_dictionary["topic_joystick_axi_control"] ?>', 'robotnik_msgs/ptz', twist_camera);
        //}
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
        //console.log(zoom_value);
        zoom_value =  -100;
        
        
        var twist_camera = {
            pan:0.0,
            tilt:0.0,
            zoom:zoom_value,
            relative:true
        };  
       publish('<?php echo $robot_dictionary["topic_joystick_axi_control"] ?>', 'robotnik_msgs/ptz', twist_camera);

    }


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
            var max_inc = 0.01;
            var x = (joystick_camera_control.deltaY()) / 10000;
            if (x > max_inc) {
                x = max_inc;
            } else if (x < -max_inc) {
                x = -max_inc;
            }
            var y = (-joystick_camera_control.deltaX()) / 10000;
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

    }, 1 / 30 * 500);


    </script>
        
    </div>
</div>

<script>

</script>
