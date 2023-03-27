<div class="backgroundPageWeb" style="margin-top:60px">
    <div class="container">
        <div class="row" id="divControlPanel">
            <div class="panel panel-default">
                <div class="panel-heading">Control</label></div>
                <div class="panel-body">
                    <div class="row">
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divGPSVisualization">
                            <div class="panel panel-danger">
                                <div class="panel-heading">Robot in map</label></div>
                                <div class="panel-body">
                                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="div3DVisualization">
                                        <div id="map" style="height: 500px;width: 100%"></div>
                                        
                                    </div>

                                </div>
                            </div>
                        </div>
                        <!--div class="col-xs-6 col-ms-6 col-md-6 col-xl-6" id="divCameraVisualization">
                            <div class="panel panel-danger">
                                <div class="panel-heading">Camera</label></div>
                                <div class="panel-body">
                                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divCameraVisualization">
                                        <div id="camera">
                                            <img src="camera_example.jpg" class="img-responsive"/>
                                        </div>
                                        
                                    </div>

                                </div>
                            </div>
                        </div-->
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divRobotMoveControl">
                            <div class="panel panel-danger">
                                <div class="panel-heading"><label><?php echo $lang['MONITOR_ROBOT_MOVE_CONTROL'] ?></label></div>
                                <div class="panel-body">
                                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divControlPanelBody">
                                        <center>
                                            <label class="switch">
                                                <input type="checkbox" id = "cbActiveJoyStick">
                                                <span class="slider2 round"></span>
                                            </label>
                                        </center>
                                    </div>
                                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divControlPanelBody">


                                        <div id="divJoystick" class="panel-body" >
                                            <img src="image/JoyStick.png"  id= "imgJoystick" class="img-responsive rounded-circle center" style="width:30%"/>
                                        </div>

                                    </div>
                                </div>
                            </div>

                        </div>
                        
                    </div>
                </div>
            </div>
        </div>
        <div class="row" id="divStatusSummary">
            <div class="panel panel-default">
                <div class="panel-heading">
                    <label><?php echo $lang['MONITOR_STATUS_SUMMARY'] ?></label>
                </div>
                <div class="panel-body">
                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divStatus">

                        <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="divStatusBody">
                            <div class="panel panel-danger">
                                <div class="panel-heading">
                                    <label><?php echo $lang['MONITOR_ROBOT_STATUS'] ?></label>
                                </div>
                                <div class="panel-body">
                                    <p><span class="dot" id="dotEmergencyStop"></span><label id="labelEmergencyStop"></label><p>
                                    <ul class="nav nav-tabs">
                                        <li class="active"><a data-toggle="tab" href="#divBatteryState"><?php echo $lang['MONITOR_BATTERY'] ?></a></li>
                                        <li id="liMonitorSafery" style="display: none;"><a data-toggle="tab" href="#divSafetytatus"><?php echo $lang['MONITOR_SAFETY'] ?></a></li>
                                        <li><a data-toggle="tab" href="#divSensors"><?php echo $lang['MONITOR_SENSORS'] ?></a></li>
                                        <li id="liMonitorElevator" style="display: none;"><a data-toggle="tab" href="#divElevator"><?php echo $lang['MONITOR_ELEVATOR'] ?></a></li>
                                        <li><a data-toggle="tab" href="#divMotorStatus"><?php echo $lang['MONITOR_MOTOR_STATUS'] ?></a></li>
                                        <li><a data-toggle="tab" href="#divIO"><?php echo $lang['MONITOR_IO'] ?></a></li>
                                        
                                    </ul>
                                    <div class="tab-content">
                                        <div id="divBatteryState" class="tab-pane fade in active">
                                            
					    <h3><?php echo $lang['MONITOR_BATTERY'] ?></h3>
					    <div class="row">
					    	<div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
					   		<div class="progress">
  								<div class="progress-bar" role="progressbar" id="progressBarBattery" aria-valuenow="70" aria-valuemin="0" aria-valuemax="100">
    									</label><label id="labelLevel">0</label><span id="glyphiconIsCharging" class="glyphicon glyphicon-flash"></span></p>
  								</div>
					    		</div>
							<p><label id="labelTimeCharge"></label></p>
					    	</div>
                                      		
					    	<div class="col-xs-12 col-ms-12 col-md-6 col-xl-6">
					    		<p><label><?php echo $lang['MONITOR_BATTERY_VOLTAGE'] ?>: </label><label id="labelVoltage">0</label></p>
					    	</div>
					    	<div class="col-xs-12 col-ms-12 col-md-6 col-xl-6">
                                            		<p><label><?php echo $lang['MONITOR_BATTERY_CURRENT'] ?>: </label><label id="labelCurrent">0</label></p>
					    	</div>
						<div class="col-xs-12 col-ms-12 col-md-6 col-xl-6">
                                                        <p><span class="dot" id="dotContactRelayStatus"> </span><label><?php echo $lang['MONITOR_BATTERY_BATTERY_CONTACT_SENSOR'] ?></label></p>
                                                </div>
                                                <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6">
                                                        <p><span class="dot" id="dotChargerRelayStatus"> </span><label><?php echo $lang['MONITOR_BATTERY_CHARGER_RELAY'] ?></label></p>
                                                </div>

                                            </div>
                                        </div>
                                        <div id="divElevator" class="tab-pane fade">
                                            <h3><?php echo $lang['MONITOR_ELEVATOR'] ?></h3>
                                            <div class="row">
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                    <label><?php echo $lang['MONITOR_ELEVATOR_STATE'] ?>:
                                                </div>
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                    <label id="labelElevatorState"><?php echo $lang['TXT_UNKNOWN'] ?></label>
                                                </div>
                                            </div>
                                            <div class="row">
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                    <label><?php echo $lang['MONITOR_ELEVATOR_POSITION'] ?>:
                                                </div>
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                    <label id="labelElevatorPosition"><?php echo $lang['TXT_UNKNOWN'] ?></label>
                                                </div>
                                            </div>
                                            <div class="row">
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                    <label><?php echo $lang['MONITOR_ELEVATOR_HEIGHT'] ?>:
                                                </div>
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                    <label id="labelElevatorHeight">0.0</label>
                                                </div>
                                            </div>
                                            
                                        </div>
                                        <div id="divSafetytatus" class="tab-pane fade">
                                            <h3><?php echo $lang['MONITOR_SAFETY'] ?></h3>
                                            <div class="row">
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <div class="row">
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <label><?php echo $lang['MONITOR_SAFETY_SAFETY_MODE']; ?>:</label>
                                                            </div>
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <label id="labelSafetyMode"><?php echo $lang['TXT_UNKNOWN'];?></label>
                                                            </div>
                                                        </div>
                                                </div>
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <div class="row">
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <label><?php echo $lang['MONITOR_SAFETY_CHARGING']; ?>:</label>
                                                            </div>
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <p>
                                                                <span class="dot" id="dotSafetyCharging"></span>
                                                                </p>
                                                            </div>
                                                        </div>
                                                </div>
                                            </div>
                                            <div class="row">
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <div class="row">
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <label><?php echo $lang['MONITOR_SAFETY_EMERGENCY_STOP']; ?>:</label>
                                                            </div>
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <p>
                                                                <span class="dot" id="dotSafetyEmergencyStop"></span>
                                                                </p>
                                                            </div>
                                                        </div>
                                                </div>
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <div class="row">
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <label><?php echo $lang['MONITOR_SAFETY_SAFETY_STOP']; ?>:</label>
                                                            </div>
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <p>
                                                                <span class="dot" id="dotSafetySafetyStop"></span>
                                                                </p>
                                                            </div>
                                                        </div>
                                                </div>
                                            </div>
                                            <div class="row">
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <div class="row">
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <label><?php echo $lang['MONITOR_SAFETY_SAFETY_OVERRIDED']; ?>:</label>
                                                            </div>
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <p>
                                                                <span class="dot" id="dotSafetySafetyOverride"></span>
                                                                </p>
                                                            </div>
                                                        </div>
                                                </div>
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <div class="row">
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <label><?php echo $lang['MONITOR_SAFETY_LASER_ON_STANDBY']; ?>:</label>
                                                            </div>
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <p>
                                                                <span class="dot" id="dotSafetyLaserOnStandby"></span>
                                                                </p>
                                                            </div>
                                                        </div>
                                                </div>
                                            </div>
                                            <div class="row">
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <div class="row">
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <label><?php echo $lang['MONITOR_SAFETY_CURRENT_SPEED']; ?>:</label>
                                                            </div>
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <p>
                                                                <label id="labelSafetyCurrentSpeed">0.0<?php echo $lang['TXT_M_S']?></label>
                                                                </p>
                                                            </div>
                                                        </div>
                                                </div>
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <div class="row">
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <label><?php echo $lang['MONITOR_SAFETY_SPEED_AT_SAFETY_STOP']; ?>:</label>
                                                            </div>
                                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                                <p>
                                                                <label id="labelSafetySpeedAtSafetyStop">0.0<?php echo $lang['TXT_M_S']?></label><label>
                                                                </p>
                                                            </div>
                                                        </div>
                                                </div>
                                            </div>
                                            <div class="row">
                                                <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                    <p><h3><?php echo $lang['MONITOR_SAFETY_LASER_STATUS'];?></h4></p>
                                                    <p><label><?php echo $lang['MONITOR_SAFETY_LASER_MODE'];?>:</label><label id="labelSafetyLaserMode"><?php echo $lang['TXT_UNKNOWN'];?></label>
                                                    <table class="display responsive nowrap" id="tableSafetyLaserStatus"  cellspacing="0" width="100%">
                                                        <thead>
                                                            <tr>
                                                                <th>
                                                                    <label><?php echo $lang['MONITOR_SAFETY_LASER_STATUS_NAME'] ?></label>
                                                                </th>
                                                                <th>
                                                                    <label><?php echo $lang['MONITOR_SAFETY_LASER_STATUS_DETECTING_OBSTACLE'] ?></label>
                                                                </th>
                                                                <th>
                                                                    <label><?php echo $lang['MONITOR_SAFETY_LASER_STATUS_CONTAMINATED'] ?></label>
                                                                </th>
                                                                <th>
                                                                    <label><?php echo $lang['MONITOR_SAFETY_LASER_STATUS_FREE_WARNING'] ?></label>
                                                                </th>
                                                            </tr>
                                                        </thead>
                                                    </table>
                                                </div>
                                            </div>
                                        </div>
                                        <div id="divMotorStatus" class="tab-pane fade">
                                            <h3><?php echo $lang['MONITOR_MOTOR_STATUS'] ?></h3>
                                            <div class="row">
                                                <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                    <table class="display responsive nowrap" id="tableMotors"  cellspacing="0" width="100%">
                                                        <thead>
                                                            <tr>
                                                                <th>
                                                                    <label><?php echo $lang['MONITOR_MOTOR_STATUS_ID'] ?></label>
                                                                </th>
                                                                <th>
                                                                    <label><?php echo $lang['MONITOR_MOTOR_STATUS_NAME'] ?></label>
                                                                </th>
                                                                <th>
                                                                    <label><?php echo $lang['MONITOR_MOTOR_STATUS_STATE'] ?></label>
                                                                </th>
                                                                <th>
                                                                    <label><?php echo $lang['MONITOR_MOTOR_STATUS_STATUS'] ?></label>
                                                                </th>
                                                            </tr>
                                                        </thead>
                                                    </table>
                                                </div>
                                            </div>
                                            
                                        </div>
                                        <div id="divIO" class="tab-pane fade">
                                            <h3><?php echo $lang['MONITOR_IO'] ?></h3>
                                            <p><label><?php echo $lang['MONITOR_IO_DIGITAL_INPUTS'] ?>: </label></p><div id="divDigitalInputs"></div>
                                            <p><label><?php echo $lang['MONITOR_IO_DIGITAL_OUTPUTS'] ?>: </label><p><div id="divDigitalOutputs"></div>
                                            <p><label><?php echo $lang['MONITOR_IO_ANALOG_INPUTS'] ?>: </label></br><label id="labelAnalogInputs"></label></p>
                                            <p><label><?php echo $lang['MONITOR_IO_ANALOG_OUTPUTS'] ?>: </label></br><label id="labelAnalogOututs"></label></p>
                                        </div>
                                        <div id="divSensors" class="tab-pane fade">
                                            <h3><?php echo $lang['MONITOR_SENSORS'] ?></h3>
                                            <div class="row">
                                                <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                    <table class="display responsive nowrap" id="tableSensors"  cellspacing="0" width="100%">
                                                        <thead>
                                                            <tr>
                                                                <th>
                                                                    <label><?php echo $lang['MONITOR_SENSORS_SENSOR_NAME'] ?></label>
                                                                </th>
                                                                <th>
                                                                    <label><?php echo $lang['MONITOR_SENSORS_SENSOR_STATUS'] ?></label>
                                                                </th>
                                                            </tr>
                                                        </thead>
                                                    </table>
                                                </div>
                                            </div>
                                        </div>
                                        
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="divOdometry">
                            <div class="panel panel-danger">
                                <div class="panel-heading">
                                
                                    <label><?php echo $lang['MONITOR_NAVIGATION_STATUS'] ?></label>
                                </div>
                                <div class="panel-body">
                                
                                <ul class="nav nav-tabs">
                                    <li class="active"><a data-toggle="tab" href="#divNavigationOdometry"><?php echo $lang['MONITOR_NAVIGATION_ODOMETRY'] ?></a></li>
                                        <li><a data-toggle="tab" href="#divNavigationPosition"><?php echo $lang['MONITOR_NAVIGATION_POSITION'] ?></a></li>
                                        
                                        
                                    </ul>
                                    <div class="tab-content">
                                        <div id="divNavigationOdometry" class="tab-pane fade in active">
                                            <h3><?php echo $lang['MONITOR_NAVIGATION_ODOMETRY'] ?></h3>
                                            <table class="table">
                                                <thead>
                                                    <tr>
                                                        <th>
                                                        </th>
                                                        <th>
                                                            <label><?php echo $lang['TXT_X'] ?></label>
                                                        </th>
                                                        <th>
                                                            <label><?php echo $lang['TXT_Y'] ?></label>
                                                        </th>
                                                        <th>
                                                            <label><?php echo $lang['TXT_Z'] ?></label>
                                                        </th>
                                                    </tr>
                                                </thead>
                                                <tbody>
                                                    <tr>
                                                        <th>
                                                            <label><?php echo $lang['MONITOR_ODOMETRY_POSITION'] ?></label>
                                                        </th>
                                                        <th>
                                                            <div id="odometry_position_x"><span>0.0</span></div>
                                                        </th>
                                                        <th>
                                                            <div id="odometry_position_y"><span>0.0</span></div>
                                                        </th>
                                                        <th>
                                                            <div id="odometry_position_z"><span>0.0</span></div>
                                                        </th>
                                                    </tr>
                                                    <tr>
                                                        <th>
                                                            <label><?php echo $lang['MONITOR_ODOMETRY_ORIENTATION'] ?></label>
                                                        </th>
                                                        <th>
                                                            <div id="odometry_orientation_x"><span>0.0</span></div>
                                                        </th>
                                                        <th>
                                                            <div id="odometry_orientation_y"><span>0.0</span></div>
                                                        </th>
                                                        <th>
                                                            <div id="odometry_orientation_z"><span>0.0</span></div>
                                                        </th>
                                                    </tr>
                                                </tbody>
                                            </table>
                                            <!--p><h4><?php echo $lang['MONITOR_NAVIGATION_VELOCITY'];?></h4></p>
                                                <table class="table">
                                                    <thead>
                                                        <tr>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_NAVIGATION_VELOCITY_LINEAR_X'];?><?php echo $lang['TXT_M_S'];?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_NAVIGATION_VELOCITY_LINEAR_Y'];?><?php echo $lang['TXT_M_S'];?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_NAVIGATION_VELOCITY_ANGULAR'];?><?php echo $lang['TXT_RAD_S'];?></label>
                                                            </th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <th>
                                                                <label id="navigation_velocity_x">0.0</label>
                                                            </th>
                                                            <th>
                                                                <label id="navigation_velocity_y">0.0</label>
                                                            </th>
                                                            <th>
                                                                <label id="navigation_velocity_theta">0.0</label>
                                                            </th>
                                                        <tr>
                                                    </tbody>
                                                </table!-->
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12 text-center">
                                                <button class="btn btn-warning" onclick="resetOdometry()"><?php echo $lang['MONITOR_NAVIGATION_ODOMETRY_RESET_ODOMETRY'];?></button>
                                            </div>
                                        </div>
                                        <div id="divNavigationPosition" class="tab-pane">
                                            <h3><?php echo $lang['MONITOR_NAVIGATION_POSITION'] ?></h3>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <table class="table">
                                                    <thead>
                                                        <tr>
                                                            <th>
                                                                <label>Latitude</label>
                                                            </th>
                                                            <th>
                                                                <label>Longitude</label>
                                                            </th>
                                                            <th>
                                                                <label>altitude</label>
                                                            </th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                        <tr>
                                                            <th>
                                                                <div id="navigation_position_x"><span>0.0</span></div>
                                                            </th>
                                                            <th>
                                                                <div id="navigation_position_y"><span>0.0</span></div>
                                                            </th>
                                                            <th>
                                                                <div id="navigation_position_theta"><span>0.0</span></div>
                                                            </th>
                                                        <tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                            <!--div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                <p><span class="dot" id="dotNavigationPositionReliable"></span><label><?php echo $lang['MONITOR_NAVIGATION_POSITION_RELIABLE'] ?></label></p>
                                            </div>
                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                <p><label><?php echo $lang['MONITOR_NAVIGATION_POSITION_FRAME'] ?>:</label><label id="labelNavigationPositionFrame"></label></p>
                                            </div-->
                                        </div>
                                        
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divSystemMonitor">
                            <div class="panel panel-danger">
                                <div class="panel-heading">
                                    <label><?php echo $lang['MONITOR_SYSTEM_MONITOR'] ?></label>
                                </div>
                                <div class="panel-body">
                                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divSystemMonitorBody">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="divTableCPU">
                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_CPU_USAGE'] ?></label>
                                                <table class="display responsive nowrap" id="tableCPU" cellspacing="0" width="100%">
                                                    <thead>
                                                        <tr id="trCPUTHead">
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_CPU_NAME'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_CPU_STATUS'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_CPU_SPEED'] ?></label>
                                                            </th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                    </tbody>
                                                </table>
                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_CPU_TEMPERATURE'] ?></label>
                                                <table class="display responsive nowrap" id="tableTemp" cellspacing="0" width="100%">
                                                    <thead>
                                                        <tr id="trTempTHead">
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_CPU_TEMPERATURE_NAME'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_CPU_TEMPERATURE_TEMPERATURE'] ?><?php echo $lang['TXT_CELSIUS'];?></label>
                                                            </th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                    </tbody>
                                                </table>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="divTableMemory">
                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_MEMORY_USAGE'] ?></label>
                                                <table class="display responsive nowrap" id="tableMemory" cellspacing="0" width="100%">
                                                    <thead>
                                                        <tr id="trMemoryTHead">
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_MEMORY_NAME'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_MEMORY_FREE'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_MEMORY_TOTAL'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_MEMORY_USED'] ?></label>
                                                            </th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                    </tbody>
                                                </table>
                                            </div>
                                            <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6" id="divTableHDD">
                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_HDD_USAGE'] ?></label>
                                                <table class="display responsive nowrap" id="tableHDD" cellspacing="0" width="100%">
                                                    <thead>
                                                        <tr id="trHDDTHead">
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_HDD_NAME'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_HDD_STATUS'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_HDD_SIZE'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_HDD_USE'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_HDD_AVAILABLE'] ?></label>
                                                            </th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divTableNet">
                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_NET_USAGE'] ?></label>
                                                <table class="responsive display nowrap" id="tableNet" cellspacing="0" width="100%">
                                                    <thead>
                                                        <tr id="trNetTHead">
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_NET_NAME'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_NET_STATUS'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_NET_INPUT'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_NET_OUTPUT'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_NET_RECEIVED'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_NET_TRANSMITTED'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_NET_RX_ERROR'] ?></label>
                                                            </th>
                                                            <th>
                                                                <label><?php echo $lang['MONITOR_SYSTEM_MONITOR_NET_TX_ERROR'] ?></label>
                                                            </th>
                                                        </tr>
                                                    </thead>
                                                    <tbody>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </div>
</div>

<script>
var gpsLongitude = 0.0;
var gpsLatitude = 0.0;
var gpsAltitude = 0.0;

var firstMapPosition = false;
function initMap() {
uluru = {lat: 39.5079743, lng: -0.4619901};
if(gpsLongitude != 0.0 && gpsLatitude != 0.0){
    uluru = {lat: gpsLatitude, lng: gpsLongitude};
}
map = new google.maps.Map(document.getElementById('map'), {
    zoom: 18,
    center: uluru,
                mapTypeId: 'satellite',
    mapTypeControl: false,
    mapTypeControlOptions: {
        style: google.maps.MapTypeControlStyle.HORIZONTAL_BAR,
        position: google.maps.ControlPosition.TOP_CENTER
    },
    zoomControl: false,
    zoomControlOptions: {
        position: google.maps.ControlPosition.LEFT_CENTER
    },
    scaleControl: true,
    streetViewControl: false,
    streetViewControlOptions: {
        position: google.maps.ControlPosition.LEFT_TOP
    },
    fullscreenControl: true
    });
marker = new google.maps.Marker({
    position: uluru,
    map: map
});
}

function refreshMap() {
    var newLatLng = new google.maps.LatLng(gpsLatitude, gpsLongitude);
    marker.setPosition(newLatLng);
}

function topicPointGPS (message){
    gpsLongitude = message["longitude"];
    gpsLatitude = message["latitude"];
    gpsAltitude = message["altitude"];
    $('#navigation_position_x').text(gpsLongitude.toFixed(4));
    $('#navigation_position_y').text(gpsLatitude.toFixed(4));
    $('#navigation_position_theta').text(gpsAltitude.toFixed(4));
    uluru = {lat: gpsLatitude, lng: gpsLongitude};
    $('#labelGPSLatitude').text(gpsLatitude.toFixed(7));
    $('#lableGPSLongitude').text(gpsLongitude.toFixed(7));
    $('#lableGPSAltitude').text(gpsAltitude.toFixed(7));
    if (!firstMapPosition){
        initMap();
        firstMapPosition = true;
    }
    refreshMap();
}

//subscribe('/astrobot/gps/navpvt','ublox_msgs/NavPVT',topicGPSState);
subscribe('<?php echo $robot_dictionary["topic_fix"];?>','sensor_msgs/NavSatFix', topicPointGPS);

var joystick;
$(document).ready(function () {
    joyStickZero = true;
    joystick = new VirtualJoystick({
        container: document.getElementById('imgJoystick'),
        mouseSupport: true,
    });

    setInterval(function () {
        if (joystick.deltaY() != 0 && joystick.deltaX() != 0 || !joyStickZero) {
            var twist = getStructTwist();
            var x = (-joystick.deltaY()) / 100;
            if (x > 0.5) {
                x = 0.5;
            } else if (x < -0.5) {
                x = -0.5;
            }
            var y = (-joystick.deltaX()) / 100;
            if (y > 0.5) {
                y = 0.5;
            } else if (y < -0.5) {
                y = -0.5;
            }
            twist.linear.x = x;
            if (x != 0 && y != 0)
                twist.angular.z = (Math.atan2(y, x));
            if(document.getElementById('cbActiveJoyStick').checked){
                publish('<?php echo $robot_dictionary["topic_joystick_robot_control"] ?>', 'geometry_msgs/Twist', twist);
            }
            if (joystick.deltaY() == 0 && joystick.deltaX() == 0) {
                joyStickZero = true;
            } else {
                joyStickZero = false;
            }
        }
        ;

    }, 1/10 * 1000);


    $('#tableCPU').DataTable({
        "bPaginate": false,
        "bFilter": false,
        "ordering": false,
        "info":     false,
        "columns": [
            {"data": "name"},
            {"data": "status"},
            {"data": "speed"}
        ]
    });

    $('#tableTemp').DataTable({
        "bPaginate": false,
        "bFilter": false,
        "ordering": false,
        "info":     false,
        "columns": [
            {"data": "name"},
            {"data": "temp"}
        ]
    });

    $('#tableMemory').DataTable({
        "bPaginate": false,
        "bFilter": false,
        "ordering": false,
        "info":     false,
        "columns": [
            {"data": "name"},
            {"data": "free"},
            {"data": "total"},
            {"data": "used"}
        ]
    });

    $('#tableHDD').DataTable({
        "bPaginate": false,
        "bFilter": false,
        "ordering": false,
        "info":     false,
        "columns": [
            {"data": "name"},
            {"data": "status"},
            {"data": "size"},
            {"data": "use"},
            {"data": "available"}
        ]
    });

    $('#tableNet').DataTable({
        "bPaginate": false,
        "bFilter": false,
        "ordering": false,
        "info":     false,
        "columns": [
            {"data": "name"},
            {"data": "state"},
            {"data": "input"},
            {"data": "output"},
            {"data": "received"},
            {"data": "transmited"},
            {"data": "rxerror"},
            {"data": "txerror"}
        ],
	"columnDefs": [
	    {targets: 0, responsivePriority: 1},
            {targets: 1, responsivePriority: 2},
            {targets: 2, responsivePriority: 3},
            {targets: 3, responsivePriority: 4},
            {targets: -4, responsivePriority: 10001}
	],
	"responsive": true
    });

    $('#tableMotors').DataTable({
        "responsive": true,
        "bPaginate": false,
        "bFilter": false,
        "ordering": false,
        "info":     false,
        "columns": [
            {"data": "can_id"},
            {"data": "name"},
            {"data": "state"},
            {"data": "status"}
        ]
    });

    $('#tableSensors').DataTable({
        "responsive": true,
        "bPaginate": false,
        "bFilter": false,
        "ordering": false,
        "info":     false,
        "columns": [
            {"data": "name"},
            {"data": "state"}
        ]
    });


    $.fn.dataTable.ext.errMode = 'none';
    readyOk = true;
});


// ODOMETRY
var odom_x = 0.0;
var odom_y = 0.0;
var odom_z = 0.0;
var odom_euler_x = 0.0;
var odom_euler_y = 0.0;
var odom_euler_z = 0.0;

// Odometry topic
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

});

function resetOdometry() {
    srv = {x: 0, y:0, z:0, orientation:0}
    callService('<?php echo $robot_dictionary["servic_reset_odometry"];?>', 'robotnik_msgs/set_odometry', srv ,function(ret){
        
    });
}

// System_monitor
subscribe('<?php echo $robot_dictionary["topic_system_status"]?>', 'system_monitor/Diagnostic', function (message) {
    if (!readyOk) {
        return;
    }
//CPU
    var table_CPU = $('#tableCPU').DataTable();
    var cpu = message.diagCpuUsage.status.cores;
    table_CPU.clear();
    $.each(cpu, function (index, val_each) {
        var dataCPU = {};
        dataCPU["name"] = "CPU-" + val_each["id"].toString();
        dataCPU["speed"] = (val_each["speed"]/100).toString()+ "%";
        dataCPU["status"] = val_each["status"].toString();
        table_CPU.row.add(dataCPU);
    });
    table_CPU.draw();
//TEMP
    var table_Temp = $('#tableTemp').DataTable();
    var temp = message.diagCpuTemp.status.cores;
    table_Temp.clear();
    $.each(temp, function (index, val_each) {
        var dataTemp = {};
        dataTemp["name"] = "Core-" + val_each["id"].toString();
        dataTemp["temp"] = val_each["temp"].toFixed(0).toString();
        table_Temp.row.add(dataTemp);
    });
    table_Temp.draw();
//Memory
    var table_Memory = $('#tableMemory').DataTable();
    table_Memory.clear();
    var memory = message.diagMem.status.memories;
    $.each(memory, function (index, val_each) {
        var dataMemory = {};
        dataMemory["name"] = val_each["name"].toString();
        dataMemory["total"] = val_each["total"].toFixed(0).toString() + "<?php echo $lang['TXT_MB'];?>";
        dataMemory["used"] = val_each["used"].toFixed(0).toString() + "<?php echo $lang['TXT_MB'];?>";
        dataMemory["free"] = val_each["free"].toFixed(0).toString() + "<?php echo $lang['TXT_MB'];?>";
        table_Memory.row.add(dataMemory);
    });
    table_Memory.draw();
//HDD
    var table_HDD = $('#tableHDD').DataTable();
    table_HDD.clear();
    var hdd = message.diagHdd.status.disks;
    $.each(hdd, function (index, val_each) {
        var dataHDD = {};
        dataHDD["name"] = val_each["name"].toString();
        dataHDD["status"] = val_each["status"].toString();
        dataHDD["size"] = val_each["size"].toString()+ "<?php echo $lang['TXT_GB'];?>";
        dataHDD["use"] = val_each["use"].toString()+ "<?php echo $lang['TXT_GB'];?>";
        dataHDD["available"] = val_each["available"].toString()+ "<?php echo $lang['TXT_GB'];?>";
        table_HDD.row.add(dataHDD);
    });
    table_HDD.draw();
//NET
    var table_NET = $('#tableNet').DataTable();
    table_NET.clear();
    var net = message.diagNet.status.interfaces;
    $.each(net, function (index, val_each) {
        var dataNET = {};
        dataNET["name"] = val_each["name"].toString();
        dataNET["state"] = val_each["state"].toString();
        dataNET["input"] = val_each["input"].toFixed(2).toString();
        dataNET["output"] = val_each["output"].toFixed(2).toString();
        dataNET["received"] = val_each["received"].toFixed(2).toString();
        dataNET["transmited"] = val_each["transmitted"].toFixed(2).toString();
        dataNET["rxerror"] = val_each["rxError"].toString();
        dataNET["txerror"] = val_each["txError"].toString();
        table_NET.row.add(dataNET);
    });
    table_NET.draw();
});


subscribe('<?php echo $robot_dictionary["topic_charge_manager_docking_status"];?>', 'robotnik_msgs/BatteryDockingStatusStamped', function (message) {
  if(message.status.contact_relay_status){
	$("#dotContactRelayStatus").css("background-color", "green");
  }else{
	$("#dotContactRelayStatus").css("background-color", "red");
  }
  if(message.status.charger_relay_status){
	$("#dotChargerRelayStatus").css("background-color", "green");
  }else{
	$("#dotChargerRelayStatus").css("background-color", "red");
  }
});

//Battery
subscribe('<?php echo $robot_dictionary["topic_hw_battery"]?>', 'robotnik_msgs/BatteryStatus', function (message) {
    $("#labelVoltage").text(message.voltage.toFixed(2));
    $("#progressBarBattery").css("width", message.level.toFixed(0)+"%");
    $("#labelCurrent").text(message.current.toFixed(2));
    $("#labelLevel").text(message.level.toFixed(0)+"%");
    if(message.is_charging){
	$("#progressBarBattery").attr('class', 'progress-bar progress-bar-info progress-bar-striped active');
	$("#glyphiconIsCharging").show();
	$("#labelTimeCharge").text(message.time_charging.toFixed(0) + "<?php echo $lang["MONITOR_BATTERY_MSG_CONNECTED"] ?>")
    }else{
	$("#labelTimeCharge").text(message.time_remaining.toFixed(0) + "<?php echo $lang["MONITOR_BATTERY_MSG_DISCONECTED"] ?>")
	$("#glyphiconIsCharging").hide();
	if(message.level.toFixed(0)>=0 && message.level.toFixed(0)<=15){
		$("#progressBarBattery").attr('class', 'progress-bar progress-bar-danger');
	}else if(message.level.toFixed(0)>15 && message.level.toFixed(0)<=25){
		$("#progressBarBattery").attr('class', 'progress-bar progress-bar-warning');
	}else{
		$("#progressBarBattery").attr('class', 'progress-bar progress-bar-success');
	}
    }
});

//RobotLocalControl
var first_message = true;
var has_safety = false;
var has_elevator = false;
subscribe('<?php echo $robot_dictionary["topic_robot_local_state"];?>', 'robot_local_control_msgs/Status', function (message) {
    var safety = message.robot_status.safety_status;
    var elevator = message.robot_status.elevator;
    if(first_message){
        if(elevator.state != ''){
            has_elevator = true;
            $('#liMonitorElevator').show();
        }
        if(safety.safety_mode != ''){
            has_safety = true;
            $('#tableSafetyLaserStatus').DataTable({
                "bPaginate": false,
                "bFilter": false,
                "ordering": false,
                "info":     false,
                "columns": [
                    {"data": "name"},
                    {
                        "data": "detecting_obstacles",
                        "render": function (data = false, type, full, meta){
                            if(data == true){
                                return '\
                                <span class="dot" style="background-color: green"></span>\
                                ';
                            }else{
                                return '\
                                <span class="dot" style="background-color: red"></span>\
                                ';
                            }
                          }
                    
                    },
                    {
                        "data": "contaminated",
                        "render": function (data = false, type, full, meta){
                            if(data == true){
                                return '\
                                <span class="dot" style="background-color: green"></span>\
                                ';
                            }else{
                                return '\
                                <span class="dot" style="background-color: red"></span>\
                                ';
                            }
                          }
                    },
                    {
                        "data": "free_warning",
                        "render": function (data = false, type, full, meta){
                            if(data == true){
                                return '\
                                <span class="dot" style="background-color: green"></span>\
                                ';
                            }else{
                                return '\
                                <span class="dot" style="background-color: red"></span>\
                                ';
                            }
                          }
                    }
                ]
            });
            $('#liMonitorSafery').show();
        }
    }
    var table_sensors = $('#tableSensors').DataTable();
    table_sensors.clear();
    $.each(message.robot_status.sensors, function (index, val_each) {
        var data_sensor = {};
        data_sensor["name"] = val_each.name;
        data_sensor["state"] = val_each.state;
        table_sensors.row.add(data_sensor);
    });
    table_sensors.draw(); 
    var localization = message.localization_status;
    try{
       
      $('#navigation_position_x').text(localization.pose.pose.x.toFixed(4));
      $('#navigation_position_y').text(localization.pose.pose.y.toFixed(4));
      $('#navigation_position_theta').text(localization.pose.pose.theta.toFixed(4));
      
    }catch(error){
      $('#navigation_position_x').text('nan');
      $('#navigation_position_y').text('nan');
      $('#navigation_position_theta').text('nan');
    }
    $('#labelNavigationPositionFrame').text(localization.pose.header.frame_id);
    if(localization.reliable){
        $("#dotNavigationPositionReliable").css("background-color", "green");
    }
    else{
        $("#dotNavigationPositionReliable").css("background-color", "red");
    }
    var velocity = message.robot_status.velocity;
    $('#navigation_velocity_x').text(velocity.linear_x.toFixed(4));
    $('#navigation_velocity_y').text(velocity.linear_y.toFixed(4));
    $('#navigation_velocity_theta').text(velocity.angular_z.toFixed(4));
    
    if(has_elevator){
        $('#labelElevatorState').text(elevator.state);
        $('#labelElevatorPosition').text(elevator.position);
        $('#labelElevatorHeight').text(elevator.height.toFixed(2).toString() + "<?php echo $lang['TXT_M'];?>");
    }
    if(has_safety){
        $('#labelSafetyMode').text(safety.safety_mode);
        $('#labelSafetyCurrentSpeed').text(safety.current_speed.toFixed(2).toString()+"<?php echo $lang['TXT_M_S'];?>");
        $('#labelSafetySpeedAtSafetyStop').text(safety.speed_at_safety_stop.toFixed(2).toString()+"<?php echo $lang['TXT_M_S'];?>");
        if(safety.charging){
            $("#dotSafetyCharging").css("background-color", "green");
        }
        else{
            $("#dotSafetyCharging").css("background-color", "red");
        }
        if(safety.emergency_stop){
            $("#dotSafetyEmergencyStop").css("background-color", "green");
        }
        else{
            $("#dotSafetyEmergencyStop").css("background-color", "red");
        }
        if(safety.safety_stop){
            $("#dotSafetySafetyStop").css("background-color", "green");
        }
        else{
            $("#dotSafetySafetyStop").css("background-color", "red");
        }
        if(safety.safety_overrided){
            $("#dotSafetySafetyOverride").css("background-color", "green");
        }
        else{
            $("#dotSafetySafetyOverride").css("background-color", "red");
        }
        if(safety.lasers_on_standby){
            $("#dotSafetyLaserOnStandby").css("background-color", "green");
        }
        else{
            $("#dotSafetyLaserOnStandby").css("background-color", "red");
        }
        var tableSafety = $('#tableSafetyLaserStatus').DataTable();
        tableSafety.clear();
        safety.lasers_status.forEach(function (item, index){
            var auxTableSafty = {};
            auxTableSafty["name"] = item.name;
            auxTableSafty["detecting_obstacles"] = item.detecting_obstacles;
            auxTableSafty["contaminated"] = item.contaminated;
            auxTableSafty["free_warning"] = item.free_warning;
            tableSafety.row.add(auxTableSafty);
        });
        tableSafety.draw();
    }
    first_message = false;
});


//Emergency Stop
subscribe('<?php echo $robot_dictionary["topic_hw_emergency_stop"]?>', 'std_msgs/Bool', function (message) {
    text = "False";
    if(message.data){
	$("#labelEmergencyStop").text("Emergency button active");
        $("#dotEmergencyStop").css("background-color", "red");
    }else{
	$("#labelEmergencyStop").text("Ready");
        $("#dotEmergencyStop").css("background-color", "green");
    }
});

//IO
subscribe('<?php echo $robot_dictionary["topic_hw_io"]?>', 'robotnik_msgs/inputs_outputs', function (message) {
    var text_digital_inputs = "";
    var text_digital_outputs = "";
    var text_analog_inputs = "";
    var text_analog_outputs = "";
    message.digital_inputs.forEach(function (item, index){
        if(index%10 == 0 & index!=0){
            text_digital_inputs =text_digital_inputs+'<br/>';
        }
        
        if(item==true){
            text_digital_inputs =text_digital_inputs+ '<label>'+((index < 10) ? '&nbsp;&nbsp;': '')+index.toString()+'</label><span class="dot" style="background-color: green"></span>';
        }
        else{
            text_digital_inputs =text_digital_inputs+ '<label>'+((index < 10) ? '&nbsp;&nbsp;': '')+index.toString()+'</label><span class="dot" style="background-color: red"></span>';
        }
        
    });
    message.digital_outputs.forEach(function (item, index){
        if(index%6 == 0 & index!=0){
            text_digital_outputs =text_digital_outputs+'<br/>';
        }
        if(item==true){
            text_digital_outputs =text_digital_outputs+ '<label>'+((index < 10) ? '&nbsp;&nbsp;': '')+index.toString()+'</label><span class="dot" style="background-color: green"></span>';
        }
        else{
            text_digital_outputs =text_digital_outputs+ '<label>'+((index < 10) ? '&nbsp;&nbsp;': '')+index.toString()+'</label><span class="dot" style="background-color: red"></span>';
        }
        
    });
    message.analog_inputs.forEach(function (item, index){
        text_analog_inputs = text_analog_inputs + item.toFixed(3) + " ";
    });
    message.analog_outputs.forEach(function (item, index){
        text_analog_outputs = text_analog_outputs + item.toFixed(3) + " ";
    });
    $("#divDigitalInputs").html(text_digital_inputs);
    $("#divDigitalOutputs").html(text_digital_outputs);
    $("#labelAnalogInputs").text(text_analog_inputs);
    $("#labelAnalogOututs").text(text_analog_outputs);
});

//Status
subscribe('<?php echo $robot_dictionary["topic_hw_status"]?>', 'robotnik_msgs/RobotnikMotorsStatus', function (message) {
    var table_motors = $('#tableMotors').DataTable();
    table_motors.clear();
    message.name.forEach(function (item, index){
        var data_motor = {};
        data_motor["name"] = item;
        data_motor["can_id"] = message.can_id[index];
        data_motor["state"] = message.motor_status[index].state;
        data_motor["status"] = message.motor_status[index].status;
        
        table_motors.row.add(data_motor);
    });
    table_motors.draw();
});


// called periodically to update the web elemets
function mainLoop() {
//console.log("mainloop");
// update odometry
    document.querySelector('#odometry_position_x span').innerHTML = odom_x;
    document.querySelector('#odometry_position_y span').innerHTML = odom_y;
    document.querySelector('#odometry_position_z span').innerHTML = odom_z;

    document.querySelector('#odometry_orientation_x span').innerHTML = odom_euler_x;
    document.querySelector('#odometry_orientation_y span').innerHTML = odom_euler_y;
    document.querySelector('#odometry_orientation_z span').innerHTML = odom_euler_z;
}
main_loop_update_time = 100;
$(document).ready(function () {
// Sets trigger for map_nav check 
    //autorun_function();
//call to mainLoop each x ms
    setInterval(mainLoop, main_loop_update_time);
});

</script>


<script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyCVM-OKAfzF4UcexNum8mV39ujIJAFoWpk"></script>
