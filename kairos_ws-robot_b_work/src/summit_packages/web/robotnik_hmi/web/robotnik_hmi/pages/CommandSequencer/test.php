<div class="backgroundPageWeb" style="margin-top:60px">
    <div class="container-fluid">    
        <div class="row">
            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                <div class="panel-group" id ="executingDiv">
                    <div class="panel panel-success">
                        <div class="panel-heading">
                            <h4 class="panel-title">
                                <a data-toggle="collapse" href="#executinCollapse">Executing</a>
                            </h4>
                        </div>
                        <div id="executinCollapse" class="panel-collapse collapse">
                            <div class="panel-body">
                                <div class="row">
                                
                                    <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="complexExecutingStateDiv">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Executing mission:</label><label id="complexMissionNameLabel"></label>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Current sequence in execution:</label><label id="complexMissionCurrentCommandLabel"></label>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="complexMissionNextCommandDiv">
                                                <label>Next sequences:</label><label id="complexMissionNextCommandLabel"></label>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="sequenceStateDiv">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Executing sequence:</label><label id="sequenceNameLabel"></label>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Current action in execution:</label><label id="sequenceCurrentCommandLabel"></label>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="sequenceNextCommandDiv">
                                                <label>Next action:</label><label id="sequenceNextCommandLabel"></label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- JOYSTICKS -->
            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="joysticks">
                <div class="panel panel-danger">
                    <div class="panel-heading">
                        <label><?php echo $lang['MANUAL_CONTROL'] ?></label>
                    </div>
                    <div class="panel-body">
                        <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6" id="robotJoystick">
                            <ul class="nav nav-tabs">
                                <li class="active"><a data-toggle="tab" href="#divRobotTeleop"><?php echo $lang['ROBOT_TELEOPERATION'] ?></a></li>
                            </ul>
                            <div class="tab-content" style="border-left: 1px solid #ddd; border-right: 1px solid #ddd; border-bottom: 1px solid #ddd; padding: 10px;">
                                <!-- ROBOT TELEOPERATION -->
                                <div id="divRobotTeleop" class="tab-pane fade in active">
                                    <div class="row" style="max-width: 100%; height: auto; margin-left: 0px; margin-bottom: 20px;">
                                        <!--h3 style="margin: 20px;"><?php echo $lang['ROBOT_TELEOPERATION'] ?></h3-->
                                        <!-- Robot stick enable  -->
                                        <div class="col-xl-6 col-md-6 col-ls-12 col-xs-6" style="text-align:center; top: 20px;">
                                            <label class="switch">
                                                <input type="checkbox" id = "robotStickEnable" onclick="robotStickEnable()">
                                                <span class="slider2 round"></span>
                                            </label>
                                            <hr>
                                            <p><label id="robotStickEnableText" style="white-space: nowrap"><?php echo $lang['CONTROLLER_MODE'] ?></label></p>
                                        </div>
                                        <!-- Robot pad -->
                                        <div class="col-xl-6 col-md-6 col-ls-6 col-xs-6" style="text-align:center;">
                                            <img id="robot_stick_parent" src="image/joystick-base.png" style="max-width: 100%; height: auto;"/>
                                            <div id="robot_stick" style="position: absolute; max-width: 220px; margin-top:auto; margin-bottom:auto; margin-left: auto; margin-right: auto; left: 0; right: 0; text-align: center; top:0; bottom:0;">
                                                <img src="image/joystick-red.png" style="margin: 0; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 30%;"/>
                                            </div>
                                        </div>
                                    </div>
                                    <hr>
                                    <div class="row">
                                        <div class="col-xl-12 col-md-12 col-ls-12 col-xs-12" style="text-align:center;">
                                            <div class="slidecontainer" style="width: 80%; text-align: center; margin-bottom: 32px; margin-left: auto; margin-right: auto;">
                                                <p><label id="maxVel" style="white-space: nowrap">Velocidad máxima (0.3 m/s)</label></p>
                                                <input type="range" min="0" max="0.6" step="0.01" value="0.3" class="slider" id="maxVelSlider" style="margin-top: 25px; margin-left: auto; margin-right: auto; width: 70%;">
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6" id="cameraJoystick">
                            <ul class="nav nav-tabs">
                                <li class="active"><a data-toggle="tab" href="#divCameraControl"><?php echo $lang['CAMERA_CONTROL'] ?></a></li>
                            </ul>
                            <div class="tab-content" style="border-left: 1px solid #ddd; border-right: 1px solid #ddd; border-bottom: 1px solid #ddd; padding: 10px;">
                                <!-- CAMERA CONTROL -->
                                <div id="divCameraControl" class="tab-pane fade in active">
                                    <div class="row" style="max-width: 100%; height: auto; margin-left: 0px; margin-bottom: 20px;">
                                        <!--h3 style="margin: 20px;"><?php echo $lang['CAMERA_CONTROL'] ?></h3-->
                                        <!-- Center the camera button -->
                                        <div class="col-xl-6 col-md-6 col-ls-6 col-xs-6" style="text-align:center; margin-top: 5px;">
                                            <div class="slidecontainer" style="width: 80%; text-align: center; margin-bottom: 32px; margin-left: auto; margin-right: auto;">
                                                <p><label id="zoomText" style="white-space: nowrap">Zoom (0%)</label></p>
                                                <input type="range" min="0" max="100" value="0" class="slider" id="zoomSlider" style="margin-top: 25px; margin-left: auto; margin-right: auto; width: 70%;">
                                            </div>
                                            <hr>
                                            <button type="button" class="btn btn-default btn-sm" style="margin:-50px; width: 130px; margin-bottom: auto; margin-top: auto;" onclick="centerCamera()">
                                                <span class="glyphicon glyphicon-screenshot"></span><?php echo $lang['TELEOPERATION_CENTER_THE_CAMERA'];?>
                                            </button>
                                        </div>
                                        <!-- Pantilt pad -->
                                        <div class="col-xl-6 col-md-6 col-ls-6 col-xs-6" style="text-align:center;">
                                            <img src="image/joystick-base.png" style="max-width: 100%; height: auto;"/>
                                            <div id="pantilt_stick" style="position: absolute; max-width: 220px; margin-top:auto; margin-bottom:auto; margin-left: auto; margin-right: auto; left: 0; right: 0; text-align: center; top:0; bottom:0;">
                                                <img src="image/joystick-blue.png" style="margin: 0; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 30%;"/>
                                            </div>
                                        </div>
                                    </div>
                                    <hr>
                                    <div class="row">
                                        <div class="col-xl-12 col-md-12 col-ls-12 col-xs-12" style="text-align:center;">
                                            <div class="slidecontainer" style="width: 80%; text-align: center; margin-bottom: 32px; margin-left: auto; margin-right: auto;">
                                                <p><label id="maxVelPt" style="white-space: nowrap">Velocidad máxima (0.75 rad/s)</label></p>
                                                <input type="range" min="0" max="2.0" step="0.01" value="0.75" class="slider" id="maxVelPtSlider" style="margin-top: 25px; margin-left: auto; margin-right: auto; width: 70%;">
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                <div class="panel-group" id ="errorPanelDiv">
                    <div class="panel panel-warning">
                        <div class="panel-heading">
                            <h4 class="panel-title">
                                <a data-toggle="collapse" href="#executinErrorCollapse">Last error in the execution</a>
                            </h4>
                        </div>
                        <div id="executinErrorCollapse" class="panel-collapse collapse">
                            <div class="panel-body">
                                <div class="row">
                                    <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="complexExecutingStateErrorDiv">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Last error in the mission:</label><label id="complexMissionErrorLabel"></label>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="sequenceStateErrorDiv">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Last error in the sequence:</label><label id="sequenceErrorLabel"></label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                <div class="row">
                    <div class="col-xs-0 col-ms-0 col-md-4 col-xl-4">
                    </div>
                    <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                        <select class="form-control" id="sequenceCommandSelect">
                            <option value="-1">--Choose command--</option>
                        </select>
                    </div>
                    <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                        <button type="button" class="btn btn-default btn-sm" id="createCommandSequenceButton" onclick="stateLogicMaching(2)">
                            <span class="glyphicon glyphicon-plus"></span> Create sequence
                        </button>
                        <!--button type="button" class="btn btn-default btn-sm" id="modifyCommandSequenceButton">
                            <span class="glyphicon glyphicon-pencil"></span> Modify
                        </button-->
                        <button type="button" class="btn btn-default btn-sm" id="deleteCommandSequenceButton"  onclick="deleteSequence()">
                            <span class="glyphicon glyphicon-remove-circle"></span> Delete
                        </button>
                    </div>
                </div>
            </div>
        </div>
        <hr>
        <div class="row">
            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12"  id="commandSequenceDiv">
                <div class="row" id="saveRunCancelButtonsDiv">
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-center">
                        <button class="btn btn-success" id="saveSequenceButton"  onclick="saveSequence()">Save</button>
                    </div>
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-center">
                        <button class="btn btn-primary" id="runSequenceButton" onclick="runSequence()">Execute sequence</button>
                        <button class="btn btn-danger" id="runSequenceButton" onclick="cancelSequence()">Cancel sequence</button>
                    </div>
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-center">
                        <button class="btn btn-danger" id="discardSequenceButton" onclick="stateLogicMaching(0)">Cancel</button>
                    </div>
                </div>
                <div class="panel panel-danger" id="commandSequenceDiv">
                    <div class="panel-heading">
                        <div class="form-group">
                            <label>Command sequence name: </label>
                            <label id="commandSequenceLabel" type="text"></label>
                            <input type="text" class="form-control" id="commandSequenceNameInput"></input>
                        </div>
                    </div>
                    <div class="panel-body">
                        <div clas="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <div class="row">
                                <div class="col-xs-3 col-ms-3 col-md-3 col-xl-3" style="background: #FFFFFF;">
                                    <div class="row" id="newActionDiv">
                                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="listCommandAcctionToAddDiv">
                                            <div id="listActionsWithSearchDiv">
                                                <label>Add actions:</label>
                                                <input type="text" placeholder="Search.." id="searchActionInput" onkeyup="filterFunction()">
                                                <div id="listActionsDiv">
                                                    <a>None</a>
                                                </div>
                                            </div>
                                        
                                        </div>

                                        
                                    </div>

                                    <div class="row" id="actionInfoDiv">
                                        

                                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12"  id="commandParamsDiv">
                                            <div class="panel-group">
                                            <label for="listGroup">Previous action:</label><label id="previousActionLabel"></label>
                                                <div class="panel panel-default">
                                                    <div class="panel-heading">
                                                        <h4 class="panel-title">
                                                            <label>Params of </label> <label id="paramsOfLabel"></label></a>
                                                        </h4>
                                                    </div>
                                                    <div class="panel-body">
                                                        <div class="row">
                                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="panelCommands">

                                                            </div>
                                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="commandButtonsDiv">
                                                                <div class="row">
                                                                    <div class="col-xs-12 col-ms-12 col-md-4 col-xl-4">
                                                                        <button id="addCommandInSequenceButton" class="btn btn-success" onclick="addActionInSequence()">Add</button>
                                                                    </div>
                                                                    <div class="col-xs-12 col-ms-12 col-md-4 col-xl-4">
                                                                        <button id="playCommandButton" class="btn btn-primary" onclick="playCommand()">Execute</button>
                                                                    </div>
                                                                    <div class="col-xs-12 col-ms-12 col-md-4 col-xl-4">
                                                                        <button id="discardCommandInSequenceButton" class="btn btn-danger" onclick="discardActionInSequence()">Discard</button>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>

                                                    <div id="params_elem" hidden>
                                                        <div id="StringDiv">
                                                            <div class="form-group">
                                                                <label for="textGroup">Text:</label>
                                                                <input type="text" class="form-control" id="textGroup">
                                                            </div>
                                                        </div>

                                                        <div id="IntegerNumberDiv">
                                                            <div class="form-group">
                                                                <label for="numberGroup">Integer number:</label>
                                                                <input type="number" step=1 class="form-control" id="numberGroup" onkeypress="return isNumber(event)" onpaste="return false;">
                                                            </div>
                                                        </div>

                                                        <div id="DoubleNumberDiv">
                                                            <div class="form-group">
                                                                <label for="numberGroup">Double number:</label>
                                                                <input type="number" step="any" class="form-control" id="numberGroup" onkeypress="return isFloat(event)" onpaste="return false;">
                                                            </div>
                                                        </div>
                                                        <div id="BooleanDiv">
                                                            <div class="form-check">
                                                                <label class="form-check-label" for="booleanInput">
                                                                    Default checkbox
                                                                </label>
                                                                <input class="form-check-input" type="checkbox" value="" id="booleanInput">
                                                                
                                                            </div>
                                                        </div>
                                                        <!--div id="BooleanDiv">
                                                            <label id="booleanLabel">Option</label><input id="booleanInput" type="checkbox" value="">
                                                        </div-->
                                                        <div id="ListDiv">
                                                            <div class="form-group">
                                                                <label for="listGroup">Select:</label>
                                                                <select class="form-control" id="listGroup">
                                                                    <!--option>The service does not respond</option>
                                                                    <option>2</option>
                                                                    <option>3</option>
                                                                    <option>4</option-->
                                                                </select>
                                                            </div>
                                                        </div>
                                                        <!--div class="row" id="addDiscardDeleteCommand">
                                                            <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6 text-center" id="addCommandInSequenceButton">
                                                                <button class="btn btn-success">Add</button>
                                                            </div>
                                                            <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6 text-center" id="discardCommandButton">
                                                                <button class="btn btn-success">Discard</button>
                                                            </div>
                                                            <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6 text-center" id="deleteCommandButton">
                                                                <button class="btn btn-success" onclick="deleteSequence()">Delete</button>
                                                            </div>
                                                        </div-->
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    
                                </div>

                                <div class="col-xs-9 col-ms-9 col-md-9 col-xl-9">
                                    <div class="row">
                                        <label>Sequence</label>
                                        <div id="VisualizationActionDiv" style="height: 500px">
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="panel-footer">
                        
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>

class JoystickController
    {
        constructor( stickID, maxDistance, deadzone )
        {
            this.id = stickID;
            let stick = document.getElementById(stickID);

            // location from which drag begins, used to calculate offsets
            this.dragStart = null;

            // track touch identifier in case multiple joysticks present
            this.touchId = null;
            
            this.active = false;
            this.value = { x: 0, y: 0 }; 

            let self = this;

            function handleDown(event)
            {
                self.active = true;

            // all drag movements are instantaneous
            stick.style.transition = '0s';

            // touch event fired before mouse event; prevent redundant mouse event from firing
            event.preventDefault();

                if (event.changedTouches)
                self.dragStart = { x: event.changedTouches[0].clientX, y: event.changedTouches[0].clientY };
                else
                self.dragStart = { x: event.clientX, y: event.clientY };

            // if this is a touch event, keep track of which one
                if (event.changedTouches)
                self.touchId = event.changedTouches[0].identifier;
            }
            
            function handleMove(event) 
            {
                if ( !self.active ) return;

                // if this is a touch event, make sure it is the right one
                // also handle multiple simultaneous touchmove events
                let touchmoveId = null;
                if (event.changedTouches)
                {
                    for (let i = 0; i < event.changedTouches.length; i++)
                    {
                        if (self.touchId == event.changedTouches[i].identifier)
                        {
                        touchmoveId = i;
                        event.clientX = event.changedTouches[i].clientX;
                        event.clientY = event.changedTouches[i].clientY;
                        }
                    }

                    if (touchmoveId == null) return;
                }

                const xDiff = event.clientX - self.dragStart.x;
                const yDiff = event.clientY - self.dragStart.y;
                const angle = Math.atan2(yDiff, xDiff);
                const distance = Math.min(maxDistance, Math.hypot(xDiff, yDiff));
                const xPosition = distance * Math.cos(angle);
                const yPosition = distance * Math.sin(angle);

                // move stick image to new position
                stick.style.transform = `translate3d(${xPosition}px, ${yPosition}px, 0px)`;

                // deadzone adjustment
                const distance2 = (distance < deadzone) ? 0 : maxDistance / (maxDistance - deadzone) * (distance - deadzone);
                const xPosition2 = distance2 * Math.cos(angle);
                const yPosition2 = distance2 * Math.sin(angle);
                const xPercent = parseFloat((xPosition2 / maxDistance).toFixed(4));
                const yPercent = parseFloat((yPosition2 / maxDistance).toFixed(4));
                
                self.value = { x: xPercent, y: yPercent };

                if (stickID == "pantilt_stick") {
                    var maxVelPt = document.getElementById('maxVelPtSlider').value;
                    var vel_x = {
                        data: self.value.x * maxVelPt
                    };
                    publish('<?php echo $robot_dictionary["topic_pan_vel_control"] ?>', 'std_msgs/Float64', vel_x);
                    var vel_y = {
                        data: -self.value.y * maxVelPt
                    };
                    publish('<?php echo $robot_dictionary["topic_tilt_vel_control"] ?>', 'std_msgs/Float64', vel_y);
                }
            }

            function handleUp(event) 
            {
                if ( !self.active ) return;

                // if this is a touch event, make sure it is the right one
                if (event.changedTouches && self.touchId != event.changedTouches[0].identifier) return;

                // transition the joystick position back to center
                stick.style.transition = '.2s';
                stick.style.transform = `translate3d(0px, 0px, 0px)`;

                // reset everything
                self.value = { x: 0.0, y: 0.0 };
                self.touchId = null;
                self.active = false;

                if (stickID == "pantilt_stick") {
                    var vel_x = {
                        data: self.value.x
                    };
                    publish('<?php echo $robot_dictionary["topic_pan_vel_control"] ?>', 'std_msgs/Float64', vel_x);
                    var vel_y = {
                        data: -self.value.y
                    };
                    publish('<?php echo $robot_dictionary["topic_tilt_vel_control"] ?>', 'std_msgs/Float64', vel_y);
                }
            }

            stick.addEventListener('mousedown', handleDown);
            stick.addEventListener('touchstart', handleDown);
            document.addEventListener('mousemove', handleMove, {passive: false});
            document.addEventListener('touchmove', handleMove, {passive: false});
            document.addEventListener('mouseup', handleUp);
            document.addEventListener('touchend', handleUp);

            if (stickID == "robot_stick") {
                setInterval(function () {
                    if (document.getElementById('robotStickEnable').checked) {
                        var maxVel = document.getElementById('maxVelSlider').value;
                        var cmd_vel = {
                            linear: {
                                x: -self.value.y * maxVel
                            },
                            angular: {
                                z: -self.value.x * maxVel * 1.5
                            }
                        };
                        publish('<?php echo $robot_dictionary["topic_joystick_robot_control"] ?>', 'geometry_msgs/Twist', cmd_vel);
                    }
                }, 100)
            }

        }
    }
    
    

    let command_sequences_selected_ = -1
    let current_state_machine_ = 0
    let command_sequences_information_ = {}
    let gph_
    let command_sequences_information_visual_ = []
    let handlers_type_ = []
    let selected_handlers_information_ = []
    let current_handler_select_ = {}

    // Start robot and camera controllers
    let pantiltStick = new JoystickController("pantilt_stick", 40, 0);
    let robotStick = new JoystickController("robot_stick", 40, 0);

    const options = {
        manipulation: false,
        height: "90%",
        layout: {
            hierarchical: {
            enabled: true,
            levelSeparation: 150,
            },
        },
        physics: {
            hierarchicalRepulsion: {
            nodeDistance: 150,
            },
        },
    };

    
    function stateLogicMaching(new_state){
        if(new_state == 0){
            idleState()
        }
        else if (new_state == 1)
        {
            selectCommandSequenceState()
        }
        else if (new_state == 2)
        {
            createNewCommandSequenceState()
        }
        else{
            alertify.error("Error state")
            stateLogicMaching(0)
            return
        }
        current_state_machine_ = new_state
    }

    function idleState(){
        command_sequences_selected_ = -1;
        $("#commandSequenceDiv").hide();
        $("#createCommandSequenceButton").show();
        //$("#modifyCommandSequenceButton").hide();
        $("#deleteCommandSequenceButton").hide();
        $("#listCommandAcctionToAddDiv").hide();
        $("#saveRunCancelButtonsDiv").hide();
        $("#commandParamsDiv").hide();
        $("#saveSequenceButton").hide();
        $("#cancelButton").hide();
        $("#runSequenceButton").hide();
        $("#addCommandInSequenceButton").hide();
        $("#commandSequenceNameInput").hide();
        $("#commandSequenceNameInput").val("");
        
        
        $("#commandButtonsDiv").hide();
        $("#playCommandButton").hide();
        $("#discardCommandInSequenceButton").hide();
        $("#commandSequenceLabel").hide();
        getCommandSequences();
        getHandlers();
    }

    function selectCommandSequenceState(){
        $("#createCommandSequenceButton").hide();
        //$("#modifyCommandSequenceButton").hide();
        $("#deleteCommandSequenceButton").show();
        $("#listCommandAcctionToAddDiv").hide();
        $("#commandSequenceLabel").show();
        $("#commandSequenceLabel").text(command_sequences_selected_);
        chargeCommandSequence(command_sequences_selected_);
        $("#commandSequenceDiv").show();
        $("#saveSequenceButton").hide();
        $("#discardSequenceButton").hide();
        $("#runSequenceButton").show();
        $("#commandParamsDiv").hide();
        $("#saveRunCancelButtonsDiv").show();
        $("#commandButtonsDiv").hide();
        $("#playCommandButton").hide();
        $("#discardCommandInSequenceButton").hide();
        $("#commandSequenceNameInput").hide();
        $("#commandSequenceNameInput").val("");

    }

    function createNewCommandSequenceState(){

        $("#saveRunCancelButtonsDiv").show();

        $("#createCommandSequenceButton").hide();
        //$("#modifyCommandSequenceButton").hide();
        $("#deleteCommandSequenceButton").hide();
        $("#commandSequenceLabel").hide();
        $("#commandSequenceLabel").text("");
        initVisualization();
        chargeListOfHandlers();
        $("#commandSequenceDiv").show();

        $("#commandSequenceNameInput").show();
        $("#commandSequenceNameInput").val("");

        $("#listCommandAcctionToAddDiv").show();
        $("#saveSequenceButton").show();
        $("#discardSequenceButton").show();
        $("#runSequenceButton").hide();
        
    }

    function chargeListOfHandlers(){
        var list_actions_element = document.getElementById("listActionsDiv");
        list_actions_element.innerHTML = "";
        jQuery.each(handlers_type_, function( handler_type_index, handler_type_foreach ){
            var button_element = document.createElement('button');
            button_element.id="action" + handler_type_foreach["name"] + "Button";
            button_element.onclick=function(){
                chargeParamPanel(handler_type_foreach["name"], "Init", true);
            }
            button_element.className = "btn btn-block";
            button_element.textContent =  handler_type_foreach["name"];
            list_actions_element.appendChild(button_element);
        });
    }

    function discardActionInSequence(){
        $("#commandParamsDiv").hide();
        $("#listCommandAcctionToAddDiv").show();
    }

    function addActionInSequence(){
        var command = "";
        command = $("#paramsOfLabel").text();
        var i;
        handler = getTypesHandler(command);
        size_type_handler = handler.length;
        for(i=0; i<size_type_handler; i++){
            var element = $("#value_command_01_____"+i.toString());
	    if(element.val() === ""){
     		element.val("None");
	    }
            if(handler[i].type=="bool"){
                command = command + " " +element[0].checked.toString();
            }else{
                command = command + " " +element.val();
            }
        }
        addElementInVisualization(command)
        $("#commandParamsDiv").hide();
        $("#listCommandAcctionToAddDiv").show();
    }

    function initVisualization(){
        if(gph_ != null){
            gph_.destroy();
        }
        container = document.getElementById("VisualizationActionDiv");
        
        command_sequences_information_visual_ = { 
            nodes: [
                {
                    id: "Init_01_____0",
                    size: 100,
                    label: "Init",
                    color: "#0000FF",
                    shape: "cicle",
                    font: { face: "monospace", align: "left" }
                }
            ],
            edges: [

            ]
        }
        selected_handlers_information_ = [];
        gph_ = new vis.Network(container, command_sequences_information_visual_, options);
        gph_.on("click", function(params){
            
            index_handler_select = parseInt(params["nodes"][0].split("_01_____")[1]);
            if(index_handler_select == 0){
                $("#commandParamsDiv").hide();
            }else{
                current_handler_select_ = selected_handlers_information_[index_handler_select-1];
                if(current_state_machine_==1){
                    $("#commandButtonsDiv").show();
                    $("#playCommandButton").show();
                    $("#discardCommandInSequenceButton").hide();
                    chargeParamPanel(current_handler_select_["handler"], current_handler_select_["previous_action"] ,false, current_handler_select_["values"]);
                }

                $("#commandParamsDiv").show();
            }

        });
    }

    function addElementInVisualization(command){
        last_index = (command_sequences_information_visual_.nodes.length) -1;
        id_command = "";
        label_in_node = "";
        new_edge = {};
        types = [];
        handler_information = {
            handler: "",
            values: [],
            previous_action: ""
        }
            
        jQuery.each(command.split(" "), function (index_split_foreach, command_split_foreach)
        {
            if(index_split_foreach==0){
                handler_information["handler"] = command_split_foreach;
                label_in_node = command_split_foreach + " \n\n ";
                id_command = command_split_foreach +"_01_____"+(last_index+1).toString();
                new_edge = {
                    from: command_sequences_information_visual_.nodes[last_index]["id"],
                    to: id_command,
                    arrows: "to",
                    physics: false,
                    smooth: { type: "cubicBezier" }
                }
                handler_information["previous_action"] = command_sequences_information_visual_.nodes[last_index]["id"].split("_01_____")[0];
                types = getTypesHandler(command_split_foreach);
                if(types === -1){
                    return false;
                }
            }else{
                handler_information["values"].push(command_split_foreach);
                label_in_node = label_in_node + types[index_split_foreach-1]["description"]+ ": " + command_split_foreach + " \n ";
            }
        });
        selected_handlers_information_.push(handler_information);
        
        new_node = {
           id: id_command,
           size: 400,
           color: "#FFCFCF",
           shape: "box",
           font: { face: "monospace", align: "left" },
           label: label_in_node
        }
        

        command_sequences_information_visual_["nodes"].push(new_node)
        command_sequences_information_visual_["edges"].push(new_edge)
        gph_.setData(command_sequences_information_visual_);
        return true;
    }

    function chargeCommandSequence(id){
        initVisualization();
        callService('/robot/command_sequencer/get_sequences', 'robot_simple_command_manager_msgs/GetCommandManagerList', {}, function (res) {
            if(res["success"] != true){
                alertify.error("Error to charge command sequences. Is it the command sequences running?");
                return;
            }
            else{
                jQuery.each(res["list"]["commands"], function( i, list ) {
                    if(list["id"]==id){
                        list["command"]=list["command"].replaceAll(" ; ", ";");
                        list["command"]=list["command"].replaceAll("; ", ";");
                        list["command"]=list["command"].replaceAll(" ;", ";");
                        list["command"].split(";").forEach(function(command_foreach)
                        {
                            if(!addElementInVisualization(command_foreach)){
                                deleteCommandSequence(id);
                            }
                        });
                    }
                });
            }
        });
    }

    function getCommandSequences(){
        callService('/robot/command_sequencer/get_sequences', 'robot_simple_command_manager_msgs/GetCommandManagerList', {}, function (res) {
            command_sequences_selected_ = -1;
            $('#sequenceCommandSelect').empty();
            $('#sequenceCommandSelect').append($('<option>', {
                value: -1,
                text: "--Choose command--"
                }));
            $('[name=options]').val(-1);
            if(res.success != true){
                alertify.error("Error to charge command sequences. Is it the command sequences running?");
                return;
            }
            else{
                jQuery.each(res["list"]["commands"], function( i, val ) {
                    $('#sequenceCommandSelect').append($('<option>', {
                        value: val["id"],
                        text: val["id"]
                    }));
                });
            }
        });
    }

    $("#sequenceCommandSelect" ).change(function () {
        change_command_sequence_selected = false
        $( "#sequenceCommandSelect option:selected" ).each(function() {
            change_command_sequence_selected = ( $( this ).val() != command_sequences_selected_)
            command_sequences_selected_ = $( this ).val();
        });
        if(!change_command_sequence_selected){
	    return;
        }
        if(command_sequences_selected_ == -1){
            stateLogicMaching(0);
        }else{
            stateLogicMaching(1);
        }
    }).change();

    function getHandlers(){
        //handlers_type_ =[]
        callService('/robot/command_manager/get_handlers', 'robot_simple_command_manager_msgs/GetHandlers', {}, function (res) {
            handlers_type_ =[];
            console.log(res);
            jQuery.each(res["handlers"], function( handler_name_foreach_index, handler_name_foreach ){
                srv = {
                    name: handler_name_foreach
                }
                callService('/robot/command_manager/get_handler_info', 'robot_simple_command_manager_msgs/GetHandlerInfo', srv, function (get_handler_info_res) {
                    if(!get_handler_info_res["success"]){
                        alertify.error("Error to get handlers info");
                        return;
                    }else if(get_handler_info_res["description"].length!=get_handler_info_res["types"].length){
                        alertify.error("Error to charge handler " + handler_name_foreach + ". The number of types and descriptions are different.");
                        return;
                    }
                    
                    handler = {
                        name: handler_name_foreach,
                        values: []
                    }
                    jQuery.each(get_handler_info_res["description"], function( handler_name_foreach_index, handler_name_foreach ){
                        values = {
                            description: get_handler_info_res["description"][handler_name_foreach_index],
                            type: get_handler_info_res["types"][handler_name_foreach_index] 
                        }
                        handler["values"].push(values);
                    });
                    handlers_type_.push(handler);
                });
            });
        });
    }

    function getTypesHandler(handler){
        for(i = 0; i < handlers_type_.length; i++){
            if(handler == handlers_type_[i]["name"]){
                return handlers_type_[i]["values"];
            }
        }
        alertify.error("The handler " + handler + " doesn't exist");
        return -1;
    }

    let current_map = ""
    subscribe('/robot/map_nav_manager/state', 'map_nav_manager/State', function (message) {
        current_map = message.map
    });

    function chargeTagList(id_elem){
        if(current_map == ""){

        }else{
            srv = {
                environment: current_map
            }
            callService('/robot/poi_manager/get_poi_list', 'poi_manager_msgs/GetPOIs', srv, function (res) {
                $('#'+id_elem).empty();
                $('#'+id_elem).append($('<option>', {
                    value: "",
                    text: "--Choose--"
                }));
                res["p_list"].forEach(function(val) {
                    $('#'+id_elem).append($('<option>', {
                        value: val.name,
                        text: val.name
                    }));
                })
            });
        }
    }

    

    function chargeParamPanel(type_of_command, previous_action="Init", edit=false, values=[]){
        $("#listCommandAcctionToAddDiv").hide();
        $("#playCommandButton").show();
        $("#paramsOfLabel").text(type_of_command);
        if(edit){
            $("#addCommandInSequenceButton").show();
            $("#discardCommandInSequenceButton").show();
            previous_action_id = command_sequences_information_visual_["nodes"][command_sequences_information_visual_.nodes.length-1]["id"].split("_01_____")[0];
            $("#previousActionLabel").text(previous_action_id);
        }else{
            $("#addCommandInSequenceButton").hide();
            $("#discardCommandInSequenceButton").hide();
            $("#previousActionLabel").text(previous_action);
        }
        handler = getTypesHandler(type_of_command)
        if(values.length != handler.length){
            values=[];
            jQuery.each(handler, function( handler_type_index, handler_type_foreach ){
                if(type_of_command.includes("GOTO_TAG") && handler_type_foreach["description"].includes("environement")){
                    values.push(current_map);
                }
                else if(handler_type_foreach["type"]=="float"){
                    values.push(0.0);
                }else if(handler_type_foreach["type"]=="integer"){
                    values.push(0);
                }else if(handler_type_foreach["type"]=="bool"){
                    values.push(false);
                }else if(handler_type_foreach["type"]=="str"){
                    values.push("");
                }else{
                    values.push("");
                }
            });
        }
        panel_commands = document.getElementById("panelCommands")
        panel_commands.innerHTML = "";
        var element;
        var i;
        for(i=0; i < handler.length; i++){
            if(type_of_command.includes("GOTO_TAG") && handler[i]["description"].includes("tag")){
                element = document.getElementById("ListDiv").cloneNode(true);
                element.id = "elem_01_____" + i;
                element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                element.children[0].children[0].textContent=handler[i]["description"]+":";
                element.children[0].children[1].id = "value_command_01_____"+i;
                element.children[0].children[1].value="";
                element.children[0].children[1].readOnly=!edit;
                chargeTagList(element.children[0].children[1].id);
            }else if(type_of_command.includes("GOTO_TAG") && handler[i]["description"].includes("environement")){
                element = document.getElementById("StringDiv").cloneNode(true);
                element.id = "elem_01_____" + i;
                element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                element.children[0].children[0].textContent=handler[i]["description"]+":";
                element.children[0].children[1].id = "value_command_01_____"+i;
                element.children[0].children[1].value=values[i];
                element.children[0].children[1].readOnly=true;
            }else{
                if(handler[i]["type"]=="float"){
                    element = document.getElementById("DoubleNumberDiv").cloneNode(true);
                    element.id = "elem_01_____" + i;
                    element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                    element.children[0].children[0].textContent=handler[i]["description"]+":";
                    element.children[0].children[1].id = "value_command_01_____"+i;
                    element.children[0].children[1].value=values[i];
                    element.children[0].children[1].readOnly=!edit;
                }else if(handler[i]["type"]=="integer"){
                    element = document.getElementById("IntegerNumberDiv").cloneNode(true);
                    element.id = "elem_01_____" + i;
                    element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                    element.children[0].children[0].textContent=handler[i]["description"]+":";
                    element.children[0].children[1].id = "value_command_01_____"+i;
                    element.children[0].children[1].value=values[i];
                    element.children[0].children[1].readOnly=!edit;
                }else if(handler[i]["type"]=="bool"){
                    element = document.getElementById("BooleanDiv").cloneNode(true);
                    /*
                    element.children[0].textContent = handler[i]["description"];
                    element.children[0].children[0].id = "value_command_01_____"+i;
                    element.children[0].children[0].checked = values[i];
                    element.children[0].children[0].readOnly=!edit;
                    */
                    element.id = "elem_01_____" + i;
                    element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                    element.children[0].children[0].textContent=handler[i]["description"]+":";
                    element.children[0].children[1].id = "value_command_01_____"+i;
                    element.children[0].children[1].checked =(values[i]=="true");
                    element.children[0].children[1].disabled=!edit;

                }else if(handler[i]["type"]=="str"){
                    element = document.getElementById("StringDiv").cloneNode(true);
                    element.id = "elem_01_____" + i;
                    element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                    element.children[0].children[0].textContent=handler[i]["description"]+":";
                    element.children[0].children[1].id = "value_command_01_____"+i;
                    element.children[0].children[1].value=values[i];
                    element.children[0].children[1].readOnly=!edit;
                }else{
                    element = document.getElementById("StringDiv").cloneNode(true);
                    element.id = "elem_01_____" + i;
                    element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                    element.children[0].children[0].textContent=handler[i]["description"]+":";
                    element.children[0].children[1].id = "value_command_01_____"+i;
                    element.children[0].children[1].value=values[i];
                    element.children[0].children[1].readOnly=!edit;
                }
            }
            panel_commands.appendChild(element);
        }
        $("#commandButtonsDiv").show();
        $("#commandParamsDiv").show();
    }

    function playCommand(){
        alertify.confirm('Play command', 'Do you want start the command?', function(){executeCurrentCommand() }, function(){alertify.success('Discarted')});
    }

    function executeCurrentCommand(){
        command = "";
        if(current_handler_select_["handler"] != undefined){
            command = command + current_handler_select_["handler"] + " ";
            jQuery.each(current_handler_select_["values"], function (index_value_foreach, value_foreach)
            {
                console.log("-->" + command)
                command = command + value_foreach + " ";
            });
        }else{
            command = command = $("#paramsOfLabel").text() + " ";
            handler = getTypesHandler($("#paramsOfLabel").text());
            size_type_handler = handler.length;
            for(i=0; i<size_type_handler; i++){
                var element = $("#value_command_01_____"+i.toString());
		if(element.val() === ""){
			element.val("None");
		}
                if(handler[i].type=="bool"){
                    command = command + " " +element[0].checked.toString();
                }else{
                    command = command + " " +element.val();
                }
            }
            
        }
        
        srv_command = {
            command: command
        }
        console.log(srv_command);
        callService('/robot/command_manager/command', 'robot_simple_command_manager_msgs/SetCommandString', srv_command, function (res) {
            if(res["ret"]["success"]){
                alertify.success('Exectuting command'); 
            }else{
                alertify.error(res["ret"]["message"] + ". Code: " + res["ret"]["code"]); 
            }

        });
    }

    function runSequence(){
        alertify.confirm('Run sequence', 'Do you want run the sequence?', function(){executeSequence() }, function(){alertify.success('Discarded')});
    }

    function getSequenceCommand(){
        var command = "";
        var sequence = command_sequences_information_visual_["nodes"];
        jQuery.each(sequence, function(sequence_index, sequence_foreach ){
            if(sequence_index!=0){
                var type_command = sequence_foreach["id"].split("_01_____")[0];
                command = command + type_command;
                var values = sequence_foreach["label"];
                values = values.replaceAll(": ", ":");
                values = values.split("\n\n")[1];
                values = values.split(" \n ");
                var i;
                for(i=0; i<=values.length-2; i++){
                    command = command + " " +values[i].split(":")[1];
                }
                command = command + " ; ";

            }
        });
        return command.slice(0, -3);
    }

    function deleteSequence(){
        alertify.confirm('Delete sequence', 'Do you want delete the sequence?', function(){deleteCurrentSequence() }, function(){});
    }

    function deleteCurrentSequence(){
        if(current_state_machine_==1){
            var delete_sequence_srv = {
                Command: {
                    id: $("#commandSequenceLabel").text(),
                    command: ""
                }
            };

            callService('/robot/command_sequencer/remove_sequence', 'robot_simple_command_manager_msgs/ManageCommandManager', delete_sequence_srv, function (res) {
                if(res["ret"]["success"]){ 
                    alertify.success('Removed: ' + $("#commandSequenceNameInput").val());
                    stateLogicMaching(0);
                }else{
                    alertify.error("Error, to save sequence. Message: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
                }
            });
        }
    }

    function saveSequence(){
        if($("#commandSequenceNameInput").val() == ""){
            alertify.error("It is necessary to give a name to the sequence");
            return;
        }
        var command = getSequenceCommand();
        if(command_sequences_information_visual_["nodes"].lenght == 1 || command == ""){
            alertify.error("It is necessary to add at least one action");
            return;
        }
        if(current_state_machine_==2){
            var create_sequence_srv = {
                Command: {
                    id: $("#commandSequenceNameInput").val(),
                    command: command
                }
            };
            callService('/robot/command_sequencer/add_sequence', 'robot_simple_command_manager_msgs/ManageCommandManager', create_sequence_srv, function (res) {
                if(res["ret"]["success"]){ 
                    alertify.success('Saved: ' + $("#commandSequenceNameInput").val());
                    stateLogicMaching(0);
                }else{
                    alertify.error("Error, to save sequence. Message: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
                }
            });
        }else{
            alertify.error("Error, current state machine is not exist"); 
        }
    }

    function executeSequence(){
        if(current_state_machine_ == 1){
            command = $("#commandSequenceLabel").text();
            srv_command = {
                command: command
            }
            callService('/robot/command_sequencer/command', 'robot_simple_command_manager_msgs/SetCommandString', srv_command, function (res) {
                if(res["ret"]["success"]){
                    alertify.success('Exectuting sequence'); 
                }else{
                    alertify.error(res["ret"]["message"] + ". Code: " + res["ret"]["code"]); 
                }

            });
        }else{
            alertify.error("Error, current state machine is not exist"); 
        }
    }

    function cancelSequence(){
        srv = {};
        callService('/robot/command_sequencer/cancel', 'std_srvs/Trigger', srv, function (res) {
            if(res["success"]){
                alertify.success('Cancelled mission');
            }else{
                alertify.error("Error to cancel mission. Error: " + res["message"]);
                return;
            }
        });
    }

    function deleteCommandSequence(id_command_sequence){
        
    }

    function filterFunction() {
        var input, filter, ul, li, a, i;
        input = document.getElementById("searchActionInput");
        filter = input.value.toUpperCase();
        div = document.getElementById("listActionsDiv");
        a = div.getElementsByTagName("button");
        for (i = 0; i < a.length; i++) {
            txtValue = a[i].textContent || a[i].innerText;
            if (txtValue.toUpperCase().indexOf(filter) > -1) {
            a[i].style.display = "";
            } else {
            a[i].style.display = "none";
            }
        }
    }

    function isNumber(evt) {
        evt = (evt) ? evt : window.event;
        var charCode = (evt.which) ? evt.which : evt.keyCode;
        if ( (charCode > 31 && charCode < 48) || charCode > 57) {
            return false;
        }
        return true;
    }

    function isFloat(evt) {
        evt = (evt) ? evt : window.event;
        var charCode = (evt.which) ? evt.which : evt.keyCode;
        if (charCode == 46){
            return true;
        }
        if ((charCode > 31 && charCode < 48) || charCode > 57) {
            return false;
        }
        return true;
    }

    $(document).ready(function () {
    	stateLogicMaching(0);
    });


    subscribe('/robot/complex_sequencer/status', 'robot_simple_command_manager_msgs/SequencerStatus', function (message) {
        if(message.current_sequence.commands.length!=0){
            $("#complexMissionNameLabel").text(message.current_sequence.name);
            $("#complexMissionCurrentCommandLabel").text(message.current_sequence.current_command);
            if(message.current_sequence.command_index == message.current_sequence.commands.length || message.current_sequence.commands.length==0){
                $("#complexMissionNextCommandLabel").text("");
                $("#complexMissionNextCommandDiv").hide();
            }else{
                list_sequences = ""
                for(i = message.current_sequence.command_index; i < message.current_sequence.commands.length; i++){
                    list_sequences += message.current_sequence.commands[i] + " ";
                }
                $("#complexMissionNextCommandLabel").text(list_sequences);
                $("#complexMissionNextCommandDiv").show();
            }
            $("#complexExecutingStateDiv").show();
            $("#executingDiv").show();
        }else{
            $("#complexMissionNameLabel").text("");
            $("#complexExecutingStateDiv").hide();
            if($("#sequenceNameLabel").text()==""){
                $("#executingDiv").hide();
            }
        }
        $("#complexMissionErrorLabel").text(message.last_error);
        if(message.last_error==""){
            $("#complexExecutingStateErrorDiv").hide();
            if ($("#sequenceErrorLabel").text()==""){
                $("#errorPanelDiv").hide()
            }
        }else{
            $("#complexExecutingStateErrorDiv").show();
            $("#errorPanelDiv").show()
        }
    });

    subscribe('/robot/command_sequencer/status', 'robot_simple_command_manager_msgs/SequencerStatus', function (message) {
        if(message.current_sequence.commands.length!=0){
            $("#sequenceNameLabel").text(message.current_sequence.name);
            $("#sequenceCurrentCommandLabel").text(message.current_sequence.current_command);
            if(message.current_sequence.command_index == message.current_sequence.commands.length || message.current_sequence.commands.length==0){
                $("#sequenceNextCommandLabel").text("");
                $("#sequenceNextCommandDiv").hide();
            }else{
                list_sequences = ""
                for(i = message.current_sequence.command_index; i < message.current_sequence.commands.length; i++){
                    list_sequences += message.current_sequence.commands[i] + " ";
                }
                $("#sequenceNextCommandLabel").text(list_sequences);
                $("#sequenceNextCommandDiv").show();
            }
            $("#sequenceStateDiv").show();
            $("#executingDiv").show();
        }else{
            $("#sequenceNameLabel").text("");
            $("#sequenceStateDiv").hide();
            if($("#complexMissionNameLabel").text()==""){
                $("#executingDiv").hide();
            }
        }
        $("#sequenceErrorLabel").text(message.last_error);
        if(message.last_error==""){
            $("#sequenceStateErrorDiv").hide();
            if ($("#complexExecutingStateErrorDiv").text()==""){
                $("#errorPanelDiv").hide()
            }
        }else{
            $("#sequenceStateErrorDiv").show();
            $("#errorPanelDiv").show()
        }
    });
      
</script>
