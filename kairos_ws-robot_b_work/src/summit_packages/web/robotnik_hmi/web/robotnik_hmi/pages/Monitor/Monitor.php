<div class="backgroundPageWeb" style="margin-top:60px">
    <div class="container">
        <div class="row" id="divControlPanel">
            <div class="panel panel-default">
                <div class="panel-heading"><label><?php echo $lang['MONITOR_CONTROL_PANEL'] ?></label></div>
                <div class="panel-body">
                    <div class="row">
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="div3DMapVisualization">
                        <div class="panel panel-danger">
                          <ul class="nav nav-tabs" id="visualizationTabs">
                            <li class="active"><a data-toggle="tab" href="#3d">3D</a>

                            </li>
                            <li><a data-toggle="tab" href="#gmaps">GMaps</a>

                            </li>

                          </ul>
                          <div class="tab-content">
                            <div id="3d" class="tab-pane fade in active">
                              <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="div3DVisualization">
                                    <div class="row" id="occupancygrid_parent">
                                        <div class="panel-body" id="occupancygrid"></div>
                                    </div>

                              </div>
                            </div>
                            <div id="gmaps" class="tab-pane fade">
                              <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="gmapsVisualization">
                                  <div id="gmap" style="height: 500px;width: 100%"></div>
                              </div>
                            </div>
                          </div>
                        </div>


                        </div>
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divControlPanel">
                            <div class="panel panel-danger">
                                <div class="panel-heading"><label><?php echo $lang['MONITOR_MAP_CONTROL'] ?></label></div>
                                <div class="panel-body">
                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6" id="divControlPanelBody">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4">
                                                    <label><?php echo $lang['MONITOR_MAP_CONTROL_MAPPING'] ?></label>
                                                </div>

                                                <div class="col-xs-3 col-ms-3 col-md-3 col-xl-3">
                                                    <button type="button" class="btn btn-success" onclick="startModule()" id="button_start_module"> <span class="glyphicon glyphicon-play"></span> </button>
                                                    <button type="button" class="btn btn-danger" onclick="stopModule()" id="button_stop_module"> <span class="glyphicon glyphicon-stop"></span> </button>

                                                </div>
                                                <div class="col-xs-5 col-ms-5 col-md-5 col-xl-5">
                                                  <div class="input-group mb-3">
                                                    <select class="form-control" id="inputGroupSelectLocalizationModules">

                                                    </select>
                                                  </div>
                                                </div>

                                            </div>
                                        </div>

                                        <div class="row" id="saveMapRow">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <hr>
                                                <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4">
                                                    <label><?php echo $lang['MONITOR_MAP_CONTROL_NAME'] ?></label>
                                                </div>
                                                <div class="col-xs-8 col-ms-8 col-md-8 col-xl-8">
                                                    <div class="row">
                                                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                            <div class="input-group">
                                                                <input type="text" class="form-control" id="filename" placeholder="default_map_name">
                                                                <span class="input-group-btn">
                                                                    <button type="button" class="btn btn-warning" onclick="saveMap()" id="button_save_map"> <span class="glyphicon glyphicon-save"></span> </button>
                                                                </span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="row" id="environmentsRow">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <hr>
                                                <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4">
                                                    <label>Environments</label>
                                                </div>


                                                <div class="col-xs-8 col-ms-8 col-md-8 col-xl-8">
                                                  <div class="input-group ">
                                                    <select class="form-control" id="inputGroupSelectLocalizationEnvironments">

                                                    </select>
                                                    <span class="input-group-btn">
                                                        <button type="button" class="btn btn-success" onclick="switchEnvironment()" id="button_switch_environment"> <span class="glyphicon glyphicon-play"></span> </button>
                                                    </span>
                                                  </div>

                                                </div>

                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6" id="divLocalizationPanelBodyRight">
                                      <div id="divNavigationPosition" class="row">
                                          <div class="col-xs-3 col-ms-3 col-md-3 col-xl-3">
                                              <p><label><?php echo $lang['MONITOR_NAVIGATION_POSITION_TYPE'] ?>:</label></p> <label id="labelLocalizationType">Unknown</label>
                                          </div>
                                          <div class="col-xs-3 col-ms-3 col-md-3 col-xl-3">
                                              <p><label><?php echo $lang['MONITOR_NAVIGATION_POSITION_STATE'] ?>:</label></p> <label id="labelLocalizationState">Unknown</label>
                                          </div>
                                          <div class="col-xs-3 col-ms-3 col-md-3 col-xl-3">
                                              <p><span class="dot" id="dotNavigationPositionReliable"></span> <label><?php echo $lang['MONITOR_NAVIGATION_POSITION_RELIABLE'] ?></label></p>
                                          </div>
                                          <div class="col-xs-3 col-ms-3 col-md-3 col-xl-3">
                                              <p><label><?php echo $lang['MONITOR_NAVIGATION_POSITION_FRAME'] ?>:</label></p> <label id="labelLocalizationFrame"></label>
                                          </div>
                                          <div class="col-xs-3 col-ms-3 col-md-3 col-xl-3">
                                              <p><label>Environment: </label></p> <label id="labelLocalizationEnvironment"></label>
                                          </div>

                                          <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                              <table class="table">
                                                  <thead>
                                                      <tr>
                                                          <th>
                                                              <label><?php echo $lang['TXT_X'] ?><?php echo $lang['TXT_M'] ?></label>
                                                          </th>
                                                          <th>
                                                              <label><?php echo $lang['TXT_Y'] ?><?php echo $lang['TXT_M'] ?></label>
                                                          </th>
                                                          <th>
                                                              <label><?php echo $lang['TXT_THETA'] ?><?php echo $lang['TXT_RAD'] ?></label>
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


                                      </div>
                                   </div>
                                </div>
                            </div>

                        </div>

                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divOutputMonitor">
                            <div class="panel panel-danger">
                                <div class="panel-heading"><label><?php echo $lang['MONITOR_MAP_CONTROL_OUTPUT_MONITOR'] ?></label></div>
                                <div class="panel-body">
                                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="divOutputMonitorBody">
                                        <textarea class="noresize" id="monitor" style="min-width: 100%;max-width: 100%;min-height: 40%; max-height: 100%" rows="5" cols="42" readonly=""> </textarea>
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
                                    <li><a data-toggle="tab" href="#divNavigationGPS">GPS</a></li>


                                </ul>
                                    <div class="tab-content">
                                        <div id="divNavigationOdometry" class="tab-pane fade in active">

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
                                            <p><h4><?php echo $lang['MONITOR_NAVIGATION_VELOCITY'];?></h4></p>
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
                                                                <label id="navigation_velocity_angular_z">0.0</label>
                                                            </th>
                                                        <tr>
                                                    </tbody>
                                                </table>

                                        </div>
                                        <div id="divNavigationGPS" class="tab-pane">

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
                                                                <div id="navigation_gps_latitude"><span>0.0</span></div>
                                                            </th>
                                                            <th>
                                                                <div id="navigation_gps_longitude"><span>0.0</span></div>
                                                            </th>
                                                            <th>
                                                                <div id="navigation_gps_altitude"><span>0.0</span></div>
                                                            </th>
                                                        <tr>
                                                    </tbody>
                                                </table>
                                            </div>

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
var robot_local_control_status = null;
var joystick;
global_frame = '<?php echo $robot_dictionary["global_frame"]?>';
var viewer = null;

function initialize(){

  viewer = new ROS3D.Viewer({
                    divID : 'occupancygrid',
                    width : 800,
                    height : 600,
                    antialias : true,
                    background: '#002233',
                    cameraPose : {x: 0, y: 0, z: 20}
                  });


  addGrid();
  setTfClient(global_frame, '<?php echo $robot_dictionary["topic_tf2_web_republisher"] ?>', '<?php echo $robot_dictionary["republisher_tfs"] ?>');
  console.log('<?php echo $robot_dictionary["topic_tf2_web_republisher"] ?>')
  addRobot('<?php echo $robot_dictionary["description"] ?>');
  interactiveMarkerClient('<?php echo $robot_dictionary["topic_poi_interactive_marker"] ?>');
  setMap('<?php echo $robot_dictionary["topic_map"] ?>');
  laserScan('<?php echo $robot_dictionary["topic_2d_scan"] ?>');
}


initialize();

// callback when getting the rlc localization param
function callback_rlc_localization_param(value){
  //console.log('callback_rlc_localization_param: '+ value);
  for (i = 0; i < value.length; i++) {
    $('#inputGroupSelectLocalizationModules').append('<option value="'+value[i]+'">'+value[i]+'</option>');
  }


}

$(document).ready(function () {

    getParam('<?php echo $robot_dictionary["param_rlc_localization_components"] ?>',callback_rlc_localization_param);
    /*joyStickZero = true;
    joystick = new VirtualJoystick({
        container: document.getElementById('imgJoystick'),
        mouseSupport: true,
    });
    */

    setInterval(function () {
        /*if (joystick.deltaY() != 0 && joystick.deltaX() != 0 || !joyStickZero) {
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
        ;*/

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

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
var readyOk = false;

function switchEnvironment(){
  var srv = {environment: ""};
  srv["environment"] = $('#inputGroupSelectLocalizationEnvironments').val();

  if(robot_local_control_status != null && robot_local_control_status.localization_status.type != ''){
    // Check and update whether the switch of Environments is available
    service_ns = '<?php echo $robot_dictionary["rlc_localization_ns"]?>'+robot_local_control_status.localization_status.type+'/switch_environment';

    callService(service_ns, 'robot_local_control_msgs/SetEnvironment', srv, function (res) {
        if (res.success) {
            document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Switching Localization environment";
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            alertify.success("Localization Environments switched!");
        } else {
            document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: " + res.message;
            alertify.error("ERROR starting Localization Module!");
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        }
    });
  } else {
    document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: There's no localization module loaded";
    alertify.error("ERROR switching environment!");
  }

}


// gets and updates the list of environments
function updateLocalizationEnvironments(service_ns)
{
  //console.log('Calling service ' + service_ns)

  // Remove all the environments
  $('#inputGroupSelectLocalizationEnvironments').children().remove().end();

  callService(service_ns, 'robot_local_control_msgs/GetEnvironment', {}, function (res) {
      if (res.success) {
          document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: The localization environments have been updated " + res.environment;
          document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
          for (i = 0; i < res.environment.length; i++) {
            $('#inputGroupSelectLocalizationEnvironments').append('<option value="'+res.environment[i]+'">'+res.environment[i]+'</option>');
          }
          return true;
      } else {
          // disable control ?
          document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: " + res.message;
      }
  });

  return false;
}

function startModule() {
      var srv = {module_name: ""};
      srv["module_name"] = $('#inputGroupSelectLocalizationModules').val();

      document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Starting Localization Module " + srv["module_name"];
      document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
      callService('<?php echo $robot_dictionary["service_rlc_localization_start"]?>', 'robot_local_control_msgs/SwitchModule', srv, function (res) {
          if (res.success) {
              document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Localization Module started";
              document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
              alertify.success("Localization Module started!");
              // Check and update whether the switch of Environments is available
              service_ns = '<?php echo $robot_dictionary["rlc_localization_ns"]?>'+srv["module_name"]+'/get_environments';

              if(updateLocalizationEnvironments(service_ns) == false)
              {

              }
          } else {
              document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: " + res.message;
              alertify.error("ERROR starting Localization Module!");
              document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
          }
      });

}

/**
 * Function to stop mapping
 *
 */
function stopModule() {


        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Send message to stop the current Localization Module.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        callService('<?php echo $robot_dictionary["service_rlc_localization_stop"]?>', 'std_srv/Trigger', {}, function (res) {
            if (res.success) {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Localization Module stopped. ";
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
                alertify.success("Localization Module Stopped!");
                // Remove all the environments
                $('#inputGroupSelectLocalizationEnvironments').children().remove().end();
            } else {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: " + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
                alertify.error("ERROR stopping Localization Module!");
            }
        });


}


/**
 * Function to save map
 *
 */
function saveMap() {

        var file_name = $('#filename').val();
        var srv = {name: file_name, frame_id: global_frame};
        if (file_name == '') {
            alertify.error("ERROR!");
            document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: The name of the map is required!";
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        } else {
            document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Saving new map as " + file_name + ",  frame " + srv["frame_id"];
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            callService('<?php echo $robot_dictionary["service_rlc_localization_save_map"]?>', 'robot_local_control_msgs/SaveMap', srv, function (res) {
                if(res.success){
                  document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: the map has been saved";
                  document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
                  alertify.success("Map saved!");
                } else {
                  document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: " + res.message;
                  document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
                  alertify.error("ERROR saving the map!");

                }

            });

        }

}


/**
 * State map_nav_manager topic
 */
/*subscribe('<?php echo $robot_dictionary["service_map_nav_manager_base"]?>/state', 'map_nav_manager/State', function (message) {
    mapping = message.mapping;
    navigation = message.navigation;
    localization = message.localization;
    map_server = message.map_server;

    if (mapping) {
//document.getElementById("mapping_status").style.color = "Green";
        try {
            document.getElementById("button_save_map").disabled = false;
            document.getElementById("button_start_module").disabled = true;
            document.getElementById("button_stop_module").disabled = false;
        } catch (e) {

        }
    } else {
//document.getElementById("mapping_status").style.color = "Red";
        try {
            document.getElementById("button_save_map").disabled = true;
            document.getElementById("button_start_module").disabled = false;
            document.getElementById("button_stop_module").disabled = true;
        } catch (e) {

        }
    }
    if (navigation) {
// document.getElementById("navigation_status").style.color = "Green";
        document.getElementById("button_start_navigation").disabled = true;
        document.getElementById("button_stop_navigation").disabled = false;
    } else {
// document.getElementById("navigation_status").style.color = "Red";
        document.getElementById("button_start_navigation").disabled = false;
        document.getElementById("button_stop_navigation").disabled = true;
    }
    if (localization) {
//document.getElementById("localization_status").style.color = "Green";
        try {
            document.getElementById("button_start_localization").disabled = true;
            document.getElementById("button_stop_localization").disabled = false;
        } catch (e) {

        }

    } else {
//document.getElementById("localization_status").style.color = "Red";
        try {
            document.getElementById("button_start_localization").disabled = false;
            document.getElementById("button_stop_localization").disabled = true;
        } catch (e) {

        }

    }
    if (map_server) {
//document.getElementById("map_server_status").style.color = "Green";
        try {
            document.getElementById("button_start_map_server").disabled = true;
            document.getElementById("button_stop_map_server").disabled = false;
        } catch (e) {

        }

    } else {
//document.getElementById("map_server_status").style.color = "Red";
        try {
            document.getElementById("button_start_map_server").disabled = false;
            document.getElementById("button_stop_map_server").disabled = true;
        } catch (e) {

        }
    }


});*/



// Odometry topic
subscribe('<?php echo $robot_dictionary["topic_odometry"]?>', 'nav_msgs/Odometry', function (message) {
  // ODOMETRY
  var odom_x = 0.0;
  var odom_y = 0.0;
  var odom_z = 0.0;
  var odom_euler_x = 0.0;
  var odom_euler_y = 0.0;
  var odom_euler_z = 0.0;

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

  $('#odometry_position_x').text(odom_x);
  $('#odometry_position_y').text(odom_y);
  $('#odometry_position_z').text(odom_z);
  $('#odometry_orientation_x').text(odom_euler_x);
  $('#odometry_orientation_y').text(odom_euler_y);
  $('#odometry_orientation_z').text(odom_euler_z);

  $('#navigation_velocity_x').text(message.twist.twist.linear.x.toFixed(2));
  $('#navigation_velocity_y').text(message.twist.twist.linear.y.toFixed(2));
  $('#navigation_velocity_angular_z').text(message.twist.twist.angular.z.toFixed(2));

});


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
        dataCPU["name"] = "CPU-" + (val_each["id"]+1).toString();
        dataCPU["speed"] = (100 - val_each["idle"]).toFixed(1).toString()+ "%";
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
        dataTemp["name"] = "Core-" + (val_each["id"]+1).toString();
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



//RobotLocalControl
var first_message = true;
var has_safety = false;
var has_elevator = false;

subscribe('<?php echo $robot_dictionary["topic_robot_local_state"];?>', 'robot_local_control_msgs/Status', function (message) {
    if(!readyOk)
    {
        return;
    }
    var safety = message.robot_status.safety_status;
    var elevator = message.robot_status.elevator;
    var docking_status = message.robot_status.battery_docking_status;
    var battery = message.robot_status.battery;
    robot_local_control_status = message;
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
        var data_sensor = {name:"", state:""};
        data_sensor["name"] = val_each.name;
        data_sensor["state"] = val_each.state;
	//console.log(data_sensor);
        table_sensors.row.add(Object.assign({},data_sensor));
    });
    table_sensors.draw();

    // Localization
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
    $('#labelLocalizationFrame').text(localization.pose.header.frame_id);
    if(localization.reliable){
        $("#dotNavigationPositionReliable").css("background-color", "green");
    }
    else{
        $("#dotNavigationPositionReliable").css("background-color", "red");
    }
    $('#labelLocalizationType').text(localization.type);
    $('#labelLocalizationState').text(localization.state);
    $('#labelLocalizationEnvironment').text(localization.environment);


    if(has_elevator){
        $('#labelElevatorState').text(elevator.state);
        $('#labelElevatorPosition').text(elevator.position);
        $('#labelElevatorHeight').text(elevator.height.toFixed(2).toString() + "<?php echo $lang['TXT_M'];?>");
    }
    if(has_safety){
        $('#labelSafetyMode').text(safety.safety_mode);
        $('#labelSafetyCurrentSpeed').text(safety.current_speed.toFixed(2).toString()+"<?php echo $lang['TXT_M_S'];?>");
        // $('#labelSafetySpeedAtSafetyStop').text(safety.speed_at_safety_stop.toFixed(2).toString()+"<?php echo $lang['TXT_M_S'];?>");
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
        if(readyOk)
        {
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
    }

    // Docking status
    if(docking_status.contact_relay_status){
  	   $("#dotContactRelayStatus").css("background-color", "green");
    }else{
  	   $("#dotContactRelayStatus").css("background-color", "red");
    }
    if(docking_status.charger_relay_status){
  	   $("#dotChargerRelayStatus").css("background-color", "green");
    }else{
  	   $("#dotChargerRelayStatus").css("background-color", "red");
    }

    // Battery
    $("#labelVoltage").text(battery.voltage.toFixed(2));
    $("#progressBarBattery").css("width", battery.level.toFixed(0)+"%");
    $("#labelCurrent").text(battery.current.toFixed(2));
    $("#labelLevel").text(battery.level.toFixed(0)+"%");
    if(battery.is_charging){
    	$("#progressBarBattery").attr('class', 'progress-bar progress-bar-info progress-bar-striped active');
    	$("#glyphiconIsCharging").show();
    	$("#labelTimeCharge").text(battery.time_charging.toFixed(0) + "<?php echo $lang["MONITOR_BATTERY_MSG_CONNECTED"] ?>")
    }else{
    	$("#labelTimeCharge").text(battery.time_remaining.toFixed(0) + "<?php echo $lang["MONITOR_BATTERY_MSG_DISCONECTED"] ?>")
    	$("#glyphiconIsCharging").hide();
    	if(battery.level.toFixed(0)>=0 && battery.level.toFixed(0)<=15){
    		$("#progressBarBattery").attr('class', 'progress-bar progress-bar-danger');
    	}else if(battery.level.toFixed(0)>15 && battery.level.toFixed(0)<=25){
    		$("#progressBarBattery").attr('class', 'progress-bar progress-bar-warning');
    	}else{
    		$("#progressBarBattery").attr('class', 'progress-bar progress-bar-success');
    	}
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
    if(readyOk)
    {
        var table_motors = $('#tableMotors').DataTable();
        table_motors.clear();
        message.name.forEach(function (item, index){
            var data_motor = {};
            data_motor["name"] = item;
            data_motor["can_id"] = message.can_id[index];
            data_motor["state"] = message.motor_status[index].state;
            data_motor["status"] = message.motor_status[index].status;
        //console.log(data_motor);
            table_motors.row.add(Object.assign({},data_motor));
        });
        table_motors.draw();
    }
});

// gps (optional)
var gpsLongitude = 39.5080669;
var gpsLatitude = -0.4619233;
var gpsAltitude = 0.0;

var firstMapPosition = false;
function createMap() {

  uluru = {lat: gpsLatitude, lng: gpsLongitude};

  map = new google.maps.Map(document.getElementById('gmap'), {
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

subscribe('<?php echo $robot_dictionary["topic_fix"];?>','sensor_msgs/NavSatFix', function (message){
    gpsLongitude = message["longitude"];
    gpsLatitude = message["latitude"];
    gpsAltitude = message["altitude"];
    uluru = {lat: gpsLatitude, lng: gpsLongitude};
    $('#navigation_gps_latitude').text(gpsLatitude.toFixed(7));
    $('#navigation_gps_longitude').text(gpsLongitude.toFixed(7));
    $('#navigation_gps_altitude').text(gpsAltitude.toFixed(7));
    if (!firstMapPosition){
        createMap();
        firstMapPosition = true;
    }
    refreshMap();
});


// called periodically to update the web elemets
function mainLoop() {
//console.log("mainloop");
// update odometry

  if (robot_local_control_status != null){
    // update the list of environments
    var length = $('#inputGroupSelectLocalizationEnvironments').children('option').length;
    if(length == 0 && robot_local_control_status.localization_status.type != ''){
      // Check and update whether the switch of Environments is available
      service_ns = '<?php echo $robot_dictionary["rlc_localization_ns"]?>'+robot_local_control_status.localization_status.type+'/get_environments';

      updateLocalizationEnvironments(service_ns);
    }

    if(robot_local_control_status.localization_status.type != ''){
      $("#button_start_module").prop("disabled",true);
      $("#button_save_map").prop("disabled",false);
      $("#button_stop_module").prop("disabled",false);
      $("#inputGroupSelectLocalizationModules").val(robot_local_control_status.localization_status.type);
      $("#inputGroupSelectLocalizationModules").change();
      $("#inputGroupSelectLocalizationModules").prop("disabled",true);
      if(length != 0){
        $("#button_switch_environment").prop("disabled",false);
      }else{
        $("#button_switch_environment").prop("disabled",true);
      }
      if(robot_local_control_status.localization_status.type == 'Amcl')
      {
        $("#saveMapRow").hide();
        $("#environmentsRow").show();
        /*
        if($(inputGroupSelectLocalizationEnvironments).val() != robot_local_control_status.localization_status.environment)
        {
            $(inputGroupSelectLocalizationEnvironments).val(robot_local_control_status.localization_status.environment);
            $(inputGroupSelectLocalizationEnvironments).change();
        }
        */
      }
      else if(robot_local_control_status.localization_status.type == 'Gmapping')
      {
        $("#saveMapRow").show();
        $("#environmentsRow").hide();
      }
      else
      {
        $("#saveMapRow").hide();
        $("#environmentsRow").hide();
      }
    }else{
      $("#button_start_module").prop("disabled",false);
      $("#button_save_map").prop("disabled",true);
      $("#button_stop_module").prop("disabled",true);
      $("#button_switch_environment").prop("disabled",true);
      $("#inputGroupSelectLocalizationModules").prop("disabled",false);
      $("#saveMapRow").hide();
      $("#environmentsRow").hide();
    }

    if(robot_local_control_status.localization_status.pose.header.frame_id != '' && global_frame != robot_local_control_status.localization_status.pose.header.frame_id){
      document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: update global frame from " + global_frame + " to " + robot_local_control_status.localization_status.pose.header.frame_id;
      global_frame = robot_local_control_status.localization_status.pose.header.frame_id;
      $("#occupancygrid").remove();
      $("#occupancygrid_parent").append("<div class='panel-body' id='occupancygrid'></div>");
      initialize();
    }

  }


}

$(document).ready(function () {
    //call to mainLoop each x ms
    setInterval(mainLoop, main_loop_update_time);
});

</script>

<script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyCVM-OKAfzF4UcexNum8mV39ujIJAFoWpk"></script>
