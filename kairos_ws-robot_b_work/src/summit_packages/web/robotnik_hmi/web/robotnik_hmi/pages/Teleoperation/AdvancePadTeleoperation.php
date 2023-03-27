<div class="backgroundPageWeb" style="margin-top:60px">

    <div class="row">
        <div class=" col-xl-12 col-md-12 col-ls-12 col-xs-12">
	
			<div class="row">
				<div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
					<div class="panel panel-danger">
						<div class="panel-heading">Video preferences</div>
						<div class="panel-body">
							<div class="row">
								<div class="form-group">
								  <label for="camera_option">Choose a camera stream:</label>
								  <select class="form-control" id="scamera_optionel1">
								    <option value="192.168.0.185" select>Axis</option>
								    <option value="192.168.0.134">RGBD</option>
								  </select>
								</div>
							</div>
							<div class="row" id="params_front">
								<div class="col-xl-4 col-md-4 col-ls-6 col-xs-6">
									<label>Resolution: </label>
									<input type="radio" name="resolution" id="320x180"/>
									<label for="320x180">320x180</label>
									<input type="radio" name="resolution" id="1280x720" checked/>
									<label for="1280x720">1280x720</label>
									<input type="radio" name="resolution" id="1920x1080" />
									<label for="1920x1080">1920x1080</label>
								</div>
								<div class="col-xl-4 col-md-4 col-ls-6 col-xs-6">
									<label>Compresion:</label>
									<input type="radio" name="compression" id="compresion_0" value="0"/>
									<label for="0">0%</label>
									<input type="radio" name="compression" id="compresion_25" value="25" checked/>
									<label for="25">25%</label>
									<input type="radio" name="compression" id="compresion_50" value="50"/>
									<label for="50">50%</label>
									<input type="radio" name="compression" id="compresion_75" value="75"/>
									<label for="75">75%</label>
									<input type="radio" name="compression" id="compresion_100" value="100"/>
									<label for="100">100%</label>
								</div>
								<div class="col-xl-4 col-md-4 col-ls-12 col-xs-12">
									<label>FPS:</label>
									<input type="radio" name="fps" id="fps_2" value="2"/>
									<label for="2">2 FPS</label>
									<input type="radio" name="fps" id="fps_15" value="15" checked/>
									<label for="15">15 FPS</label>
									<input type="radio" name="fps" id="fps_30" value="30"/>
									<label for="30">30 FPS</label>
									<input type="radio" name="fps" id="fps_60" value="60"/>
									<label for="60">60 FPS</label>
									
								</div>
								
							</div>
							<div class="row" id="params_back" hidden>
								<div class="col-xl-4 col-md-4 col-ls-6 col-xs-6">
									<label>Resolution: </label>
									<input type="radio" name="resolution_back" id="320x180"/>
									<label for="320x180">320x180</label>
									<input type="radio" name="resolution_back" id="800x450" checked/>
									<label for="800x450">800x450</label>
									<input type="radio" name="resolution_back" id="800x600"/>
									<label for="800x600">800x600</label>
								</div>
								<div class="col-xl-4 col-md-4 col-ls-6 col-xs-6">
									<label>Compresion:</label>
									<input type="radio" name="compression_back" id="compresion_0" value="0"/>
									<label for="0">0%</label>
									<input type="radio" name="compression_back" id="compresion_25" value="25" checked/>
									<label for="25">25%</label>
									<input type="radio" name="compression_back" id="compresion_50" value="50"/>
									<label for="50">50%</label>
									<input type="radio" name="compression_back" id="compresion_75" value="75"/>
									<label for="75">75%</label>
									<input type="radio" name="compression_back" id="compresion_100" value="100"/>
									<label for="100">100%</label>
								</div>
								<div class="col-xl-4 col-md-4 col-ls-12 col-xs-12">
									<label>FPS:</label>
									<input type="radio" name="fps_back" id="fps_2" value="2"/>
									<label for="2">2 FPS</label>
									<input type="radio" name="fps_back" id="fps_15" value="15" checked/>
									<label for="15">15 FPS</label>
									<input type="radio" name="fps_back" id="fps_30" value="30"/>
									<label for="30">30 FPS</label>
									<input type="radio" name="fps_back" id="fps_60" value="60"/>
									<label for="60">60 FPS</label>
									
								</div>
							</div>
							<div class="row">
							<div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
									<button id="apply_ptz_configuration" class="btn btn-default">Apply</button>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col-xl-8 col-md-8 col-ls-7 col-xs-12">
					<div class="panel panel-danger">
						<div class="panel-heading">Video</div>
						<div class="panel-body">
							
							<div class="row">
								<div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
									<center>
										<img id="ptz_video" src="http://192.168.0.185/mjpg/video.mjpg?overview=0&camera=1&resolution=800x450&videoframeskipmode=empty&Axis-Orig-Sw=true" class="img-responsive" draggable="false"/>
									</center>
								</div>
								<div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
									<center>
										<button id="fullScreen" class="btn btn-primmary" onclick="videoFullScreen()"><span class="glyphicon glyphicon-resize-full"></span><?php echo $lang['TELEOPERATION_FULL_SCREEN'];?> </button>
									</center>
									<script>
										function videoFullScreen(){
											var featuredimg = document.getElementById("ptz_video")
											if (featuredimg.requestFullscreen){
												featuredimg.requestFullscreen() 
											}
										}
									</script>
								</div>
							</div>
						</div>
					</div>
				</div>
				
				<div class="col-xl-4 col-md-4 col-ls-5 col-xs-12">
					<div class="panel panel-danger">
						<div class="panel-heading">Control video</div>
						<div class="panel-body">
							<div class="row">
								<div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
									<div class="row">
										<div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
											<div class=row>
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
											<hr>
										</div>
										<div class=" col-xl-12 col-md-12 col-ls-12 col-xs-12">
											<div class=row>
												<div class=" col-xl-12 col-md-12 col-ls-12 col-xs-12">
													<center>
														<button type="button" id="axis_led_on" class="btn btn-default btn-sm" onclick="ledOn()">
															<span class="glyphicon glyphicon-eye-close"></span>LED ON
														</button>
														<button type="button" id="axis_led_off" class="btn btn-default btn-sm" onclick="ledOff()">
															<span class="glyphicon glyphicon-eye-open"></span>LED OFF
														</button>
													</center>
												</div>
											</div>
										</div>
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col-xl-4 col-md-4 col-ls-5 col-xs-12">
					<div class="panel panel-danger">
						<div class="panel-heading">Robot status</div>
						<div class="panel-body">
							<div class="row">
								<div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
									<div class="row">
										<div class="col-xl-6 col-md-6 col-ls-6 col-xs-6">
											<label>Imu</label>
											<p>
											<label>Roll:</label><label id="row_label">0</label><label> º</label>
											<p>
											<label>Pitch:</label><label id="pitch_label">0</label><label> º</label>
											<p>
										</div>
										<div class="col-xl-6 col-md-6 col-ls-6 col-xs-6">
											<label>Velocity</label>
											<p>
											<label>Linear:</label><label id="v_linear_label">0</label><label> m/s</label>
											<p>
											<label>Angular:</label><label id="v_angular_label">0</label><label> rad/s</label>
										</div>
									</div>
									<div class="row">
										<!--div class="col-xl-6 col-md-6 col-ls-6 col-xs-6">
											<label>Odom</label>
											<p>
											<label>x:</label><label id="odom_x">0</label><label> m</label>
											<p>
											<label>y:</label><label id="odom_y">0</label><label> m</label>
											<p>
											<label>theta:</label><label id="odom_theta">0</label><label> º</label>
										</div-->
										<div class="col-xl-6 col-md-6 col-ls-6 col-xs-6">
											<label>Position</label>
											<p>
											<label>x:</label><label id="slam_odom_x">0</label><label> m</label>
											<p>
											<label>y:</label><label id="slam_odom_y">0</label><label> m</label>
											<p>
											<label>theta:</label><label id="slam_odom_theta">0</label><label> º</label>
										</div>
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>

				<div class="col-xl-12 col-md-12 col-ls-12 col-xs-12">
					<div class="panel panel-danger">
						<div class="panel-heading">Debug</div>
						<div class="panel-body">
							<div class="row">
								<div class=" col-xl-12 col-md-12 col-ls-12 col-xs-12">
									<div id="gamepadPrompt"></div>
									<div id="gamepadDisplay"></div>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
        </div>
    </div>
    
    
    
    <!--script src="js/d3.v4.min.js"></script>
    <script src="js/media-stream-library.min.js"></script>
    <script src="js/media-stream-player.min.js"></script-->
 
    <script>
		var hasGP = false;
		var repGP;
		var ptz_pantilt_inc = 0.02;
		var ptz_zoom_inc = 100;
		var ptz_inc = 0.01;
		var ptz_pantilt_max_inc = 0.2;
		var ptz_pantilt_min_inc = 0.01;
		var show_buttons_log = true;
		var velocity_multiplicator = 0.5;

		// ODOMETRY
		var odom_x = 0.0;
		var odom_y = 0.0;
		var odom_z = 0.0;
		var odom_euler_x = 0.0;
		var odom_euler_y = 0.0;
		var odom_euler_z = 0.0;
		
		// Odometry topic
		/*
		subscribe('<?php echo $robot_dictionary["topic_odometry"]?>', 'nav_msgs/Odometry', function (message) {
		    odom_x = message.pose.pose.position.x.toFixed(2);
		    odom_y = message.pose.pose.position.y.toFixed(2);
		    odom_z = message.pose.pose.position.z.toFixed(2);

		// Get euler angles from quaternion
		    var quaternion = new THREE.Quaternion(message.pose.pose.orientation.x,
			    message.pose.pose.orientation.y,
			    message.pose.pose.orientation.z,
			    message.pose.pose.orientation.w);
		    var euler = new THREE.Euler().setFromQuaternion(quaternion);

		    odom_euler_x = euler.x.toFixed(2);
		    odom_euler_y = euler.y.toFixed(2);
		    odom_euler_z = euler.z.toFixed(2);
		    $("#odom_x").text(odom_x);
		    $("#odom_y").text(odom_y);
		    $("#odom_theta").text(odom_euler_z);

		});
		*/
		// ODOMETRY
		var slam_odom_x = 0.0;
		var slam_odom_y = 0.0;
		var slam_odom_z = 0.0;
		var slam_odom_euler_x = 0.0;
		var slam_odom_euler_y = 0.0;
		var slam_odom_euler_z = 0.0;
		
		// Odometry topic
		subscribe('/robot/front_rgbd_camera/odom/sample', 'nav_msgs/Odometry', function (message) {
		    slam_odom_x = message.pose.pose.position.x.toFixed(2);
		    slam_odom_y = message.pose.pose.position.y.toFixed(2);
		    slam_odom_z = message.pose.pose.position.z.toFixed(2);

		// Get euler angles from quaternion
		    var quaternion = new THREE.Quaternion(message.pose.pose.orientation.x,
			    message.pose.pose.orientation.y,
			    message.pose.pose.orientation.z,
			    message.pose.pose.orientation.w);
		    var euler = new THREE.Euler().setFromQuaternion(quaternion);

		    slam_odom_euler_x = euler.x.toFixed(2);
		    slam_odom_euler_y = euler.y.toFixed(2);
		    slam_odom_euler_z = euler.z.toFixed(2);
		    $("#slam_odom_x").text(slam_odom_x);
		    $("#slam_odom_y").text(slam_odom_y);
		    $("#slam_odom_theta").text((slam_odom_euler_z*180/Math.PI).toFixed(2));

		});
		
		
		subscribe('/robot/robotnik_base_control/cmd_vel', 'geometry_msgs/Twist', function (message) {
		    $("#v_linear_label").text(message.linear.x.toFixed(2));
		    $("#v_angular_label").text(message.angular.z.toFixed(2));

		});

	        /*subscribe('/robot/imu/data', 'sensor_msgs/Imu', function (message) {
		    var quaternion = new THREE.Quaternion(message.orientation.x,
			    message.orientation.y,
			    message.orientation.z,
			    message.orientation.w);
		    var euler = new THREE.Euler().setFromQuaternion(quaternion);
		    orientation_euler_x = euler.x.toFixed(2);
		    orientation_euler_y = euler.y.toFixed(2);
		    orientation_euler_z = euler.z.toFixed(2);
		    $("#row_label").text((orientation_euler_x*180/Math.PI).toFixed(2));
		    $("#pitch_label").text((orientation_euler_y*180/Math.PI).toFixed(2));
		    $("#yaw_label").text((orientation_euler_z*180/Math.PI).toFixed(2));

		});
		*/

		subscribe('/robot/imu/rpy/filtered', 'geometry_msgs/Vector3Stamped', function (message) {
                    /*var quaternion = new THREE.Quaternion(message.orientation.x,
                            message.orientation.y,
                            message.orientation.z,
                            message.orientation.w);
                    var euler = new THREE.Euler().setFromQuaternion(quaternion);
                    orientation_euler_x = euler.x.toFixed(2);
                    orientation_euler_y = euler.y.toFixed(2);
                    orientation_euler_z = euler.z.toFixed(2);*/
                    $("#row_label").text((message.vector.x*180/Math.PI).toFixed(2));
                    $("#pitch_label").text((message.vector.y*180/Math.PI).toFixed(2));
                    //$("#yaw_label").text((message.vector.z*180/Math.PI).toFixed(2));

                });
	
		function canGame() {
			return "getGamepads" in navigator;
		}
	
		function reportOnGamepad() {
			console.log(navigator.getGamepads());
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
			var upper_multiplicator = 3;
			var dower_multiplicator = 0;
			
		
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
						if(gp.buttons[upper_multiplicator].pressed){
							velocity_multiplicator = velocity_multiplicator + 0.1;
							if(velocity_multiplicator > 1){
								velocity_multiplicator = 1;
							}
						}else if(gp.buttons[dower_multiplicator].pressed){
							velocity_multiplicator = velocity_multiplicator - 0.1;
							if(velocity_multiplicator < 0.1){
								velocity_multiplicator = 0.1;
							}
						}

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
				}, 1/5*1000);
			}
	
		});


		function joystick (linear, angular){
			
			var twist = getStructTwist();
			twist.linear.x = linear*-1 * velocity_multiplicator;
			//QUICK HACK: Multiplied by 0.5 to reduce maximum speed to 0.5rad/s
			twist.angular.z = angular*-1 * 0.5;
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
			publish('<?php echo $robot_dictionary["topic_joystick_axi_control"] ?>', 'robotnik_msgs/ptz', twist_camera);
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
		/*joystick_camera_control = new VirtualJoystick({
			container: document.getElementById('control_ptz'),
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
		*/
		
		//NEW VIDEO
		//const { pipelines } = window.mediaStreamLibrary

		// force auth
		const authorize = async (host) => {
			// Force a login by fetching usergroup
			const fetchOptions = {
				credentials: 'include',
				headers: {
					'Axis-Orig-Sw': true,
					'X-Requested-With': 'XMLHttpRequest',
				},
				mode: 'no-cors',
			}
			try {
				await window.fetch(`http://${host}/axis-cgi/usergroup.cgi`, fetchOptions)
			} catch (err) {
				console.error(err)
			}
		}

		const play = (host, encoding = 'h264', resolution, compression, fps) => {
			const videoEl = document.querySelector('#video_rtsp')
			const canvasEl = document.querySelector('#canvas_rtsp')
			// Grab a reference to the video element
			let Pipeline
			let mediaElement
			if (encoding === 'h264') {
				console.log("H264 - Encoding --> " + encoding);
				Pipeline = pipelines.Html5VideoPipeline
				mediaElement = videoEl
				// hide the other output
				videoEl.style.display = ''
				canvasEl.style.display = 'none'
			} else {
                                console.log("Other - Encoding --> " + encoding);
				Pipeline = pipelines.Html5CanvasPipeline
				mediaElement = canvasEl
				// hide the other output
				videoEl.style.display = 'none'
				canvasEl.style.display = ''
			}

			// Setup a new pipeline
			const pipeline = new Pipeline({
				ws: { uri: `ws://${host}/rtsp-over-websocket` },
				rtsp: { uri: `rtsp://${host}/axis-media/media.amp?videocodec=${encoding}&resolution=${resolution}&clock=1&compression=${compression}&fps=${fps}&pull=0&videoframeskipmode=empty` },
				mediaElement,
			})
			pipeline.onSourceOpen = (mse, tracks) => {
				console.log("setup mse duration")
				mse.duration = 1
			}

			pipeline.ready.then(() => {
				pipeline.rtsp.play()
			})

			return pipeline
		}

		let pipeline

		async function startPlay () {
			pipeline && pipeline.close()

                        const device = document.querySelector('#device')
                        const host = "192.168.0.185"
                        const encoding = document.querySelector('input[name=encoding]:checked').id
                        const resolution = document.querySelector('input[name=resolution]:checked').id
                        const compression = document.querySelector('input[name=compression]:checked').value
                        const fps = document.querySelector('input[name=fps]:checked').value

                        console.log(host, encoding)

                        await authorize(host)
                        pipeline = play(host, encoding, resolution, compression, fps)
			first_frame = true;

		}

		let host_camera = "192.168.0.185" 
		$('#scamera_optionel1').change(function(){
               		console.log($(this).val());
                        host_camera = $(this).val();
			if(host_camera=="192.168.0.185"){
				$('#params_back').hide();
				$('#params_front').show();
				
			}else{
				$('#params_back').show();
				$('#params_front').hide();
				
			}
			applyPtzParams();
                });
		function applyPtzParams(){
			let resolution
			let compression
			let fps
			if(host_camera=="192.168.0.185"){
				resolution = document.querySelector('input[name=resolution]:checked').id
                        	compression = document.querySelector('input[name=compression]:checked').value
                        	fps = document.querySelector('input[name=fps]:checked').value
			}else{
				resolution = document.querySelector('input[name=resolution_back]:checked').id
                        	compression = document.querySelector('input[name=compression_back]:checked').value
                        	fps = document.querySelector('input[name=fps_back]:checked').value

			}
			let src = `http://${host_camera}/mjpg/video.mjpg?overview=0&camera=1&resolution=${resolution}&videoframeskipmode=empty&Axis-Orig-Sw=true&compression=${compression}&fps=${fps}`;
			$("#ptz_video").attr("src",src);
		}

		applyPtzParams();

		// Each time a device ip is entered, authorize and then play
		const playButton = document.querySelector('#apply_ptz_configuration')
		playButton.addEventListener('click', async (e) => {
			applyPtzParams();
		})

		let last_reset = new Date();
		let timerCheckDelay = setInterval(checkDelay, 500);
		let start_time; 
		let first_frame = true;
		function checkDelay() {
			if(pipeline!== undefined && pipeline.currentTime !== undefined){
				current_time_frame = pipeline.currentTime.toFixed(2);
				current_time = new Date();
				if (first_frame && current_time_frame != 0){
					console.log("Heyyyyyyy - " + current_time_frame)
					start_time = new Date(current_time);
					start_time.setMilliseconds(start_time.getMilliseconds() + current_time_frame*100);
					first_frame = false;
				}else if(first_frame){
					return;
				}
				//current_time = new Date();
				t = ((current_time - start_time)/1000) - current_time_frame;
				console.log("time: " + (current_time.getSeconds() - start_time.getSeconds()));
				console.log("pipeline: " + current_time_frame);
				console.log("t: " + t);
				$("#delay_div").show();
                                $("#delay_label").text(t.toString());
				if(t > 1.0 && (current_time - last_reset)/1000 > 3){
					console.log("reset");
					last_reset = new Date();
					startPlay();
				}
			}
		}



		/**ROS**/

		/**ROSMON**/
		let first_msg_slam = false;
		let nodes_name_slam=[];
		subscribe('/rosmon_mapping/state', 'rosmon/State', function (message) {
			console.log("heyyy");
			if(!first_msg_slam){
				$("#start_slam").show();
				$("#stop_slam").hide();
				document.getElementById("stop_slam").disabled = true;
				document.getElementById("start_slam").disabled = true;
				nodes_name_slam = [];
			}
			nodes_actives = 0;
			message.nodes.forEach((element) => {
				if(!first_msg_slam){
					nodes_name_slam.push(element.name);
				}
				if(element.state == 1){
					nodes_actives ++;
				}
			});
			if(!first_msg_slam){
				first_msg_slam = true;
			}
			if(nodes_actives == message.nodes.length){
				document.getElementById("start_slam").disabled = false;
				document.getElementById("stop_slam").disabled = false;
				$("#start_slam").hide();
				$("#stop_slam").show();
			}else if (nodes_actives == 0){
				document.getElementById("start_slam").disabled = false;
				document.getElementById("stop_slam").disabled = false;
				$("#start_slam").show();
				$("#stop_slam").hide();
			}else{
				$("#start_slam").show();
				$("#stop_slam").hide();
				document.getElementById("start_slam").disabled = true;
				document.getElementById("stop_slam").disabled = true;
			}
		});


		/**AXIS LED**/
		let axis_led_first_msg = false;
		let digital_output=1;
		subscribe('/robot/axis_io/inputs_outputs', 'robotnik_msgs/inputs_outputs', function (message) {
			if(!axis_led_first_msg){
				$("#axis_led_on").show();
				$("#axis_led_off").hide();
				document.getElementById("axis_led_on").disabled = true;
				document.getElementById("axis_led_off").disabled = true;
				axis_led_first_msg = true;
			}else{
				document.getElementById("axis_led_on").disabled = false;
				document.getElementById("axis_led_off").disabled = false;
			}
			if(!message.digital_outputs[digital_output]){
				$("#axis_led_on").show();
				$("#axis_led_off").hide();
				
			}else{
				$("#axis_led_on").hide();
				$("#axis_led_off").show();
				
			}
		});

		function ledOn(){
			if (!axis_led_first_msg){
				alertify.error("Wating for state of led");
				return;
			}
			srv_led_on = {
				output: digital_output+1,
				value: true
			}
			
			callService('/robot/axis_io/set_digital_output', 'robotnik_msgs/set_digital_output', srv_led_on, function (res) {
				axis_led_first_msg = false;
				$("#axis_led_on").hide();
				$("#axis_led_off").show();
				document.getElementById("axis_led_on").disabled = true;
				document.getElementById("axis_led_off").disabled = true;
			});
			alertify.success("Led ON");
		}

		function ledOff(){
			if (!axis_led_first_msg){
				alertify.error("Wating for state of led");
				return;
			}
			srv_led_on = {
				output: digital_output+1,
				value: false
			}
			callService('/robot/axis_io/set_digital_output', 'robotnik_msgs/set_digital_output', srv_led_on, function (res) {
				axis_led_first_msg = false;
				$("#axis_led_on").show();
				$("#axis_led_off").hide();
				document.getElementById("axis_led_on").disabled = true;
				document.getElementById("axis_led_off").disabled = true;
			});
			alertify.success("Led OFF");
		}

		/**ROS BAG**/
		let rosbag_fist_msg = false;
		subscribe('/rosbag_manager/status', 'rosbag_manager_msgs/RosbagManagerStatus', function (message) {
			if(!rosbag_fist_msg){
				$("#record_mission_button").show();
				$("#stop_record_mission_button").hide();
				document.getElementById("record_mission_button").disabled = true;
				document.getElementById("stop_record_mission_button").disabled = true;
				rosbag_fist_msg = true;
			}else{
				document.getElementById("record_mission_button").disabled = false;
				document.getElementById("stop_record_mission_button").disabled = false;
			}
			if(!message.recording){
				$("#record_mission_button").show();
				$("#stop_record_mission_button").hide();
			}else{
				$("#stop_record_mission_button").show();
				$("#record_mission_button").hide();
			}
		});

		function recordMission(start_record){
			if (!rosbag_fist_msg){
				alertify.error("Wating for state of record mission");
				return;
			}
			let action_record = "";
			if (start_record){
				action_record = "start";
			}else{
				action_record = "stop";
			}
			srv_recording = {
				action: action_record,
				name: '',
				path: ''
			}
			
			callService('/rosbag_manager/set_recording', 'rosbag_manager_msgs/Record', srv_recording, function (res) {
				rosbag_fist_msg = false;
				$("#record_mission_button").hide();
				$("#stop_record_mission_button").show();
				document.getElementById("record_mission_button").disabled = true;
				document.getElementById("stop_record_mission_button").disabled = true;
			});
			if(start_record){
				alertify.success("Recording");
			}else{
				alertify.success("Stop Recording");
			}
		}

    </script>
    
</div>

<script>

</script>
