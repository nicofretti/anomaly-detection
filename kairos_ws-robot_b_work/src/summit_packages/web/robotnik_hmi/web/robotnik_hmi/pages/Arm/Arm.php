
<div class="" style="padding:15px; margin: 2%">
    <div class="row">
        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
            <div class="container">
                <h2>Arm</h2>
                <p>Change the value of the arm joints using the slider controls.Click in the button "Send" to have the position in the arm. Click in the "Get current position" button to restaure the slider control value to the current arm position.</p>
                <form>
                    <div class="form-group">
                        <label for="joint_base">Base:</label>
                        <input type="range" id="joint_1" name="joint_base" min="-1.57" max="1.57" step="0.005">
                    </div>
                </form>
                <button class="btn btn-default" onclick="$('#joint_1').val(0)">Reset 0 </button>
                <form>
                    <div class="form-group">
                        <label for="joint_1">Joint 1:</label>
                        <input type="range" id="joint_2" name="joint_base" min="-1.57" max="1.57" step="0.005">
                    </div>
                </form>
                <button class="btn btn-default" onclick="$('#joint_2').val(0)">Reset 0 </button>
                <form>
                    <div class="form-group">
                        <label for="joint_2">Joint 2:</label>
                        <input type="range" id="joint_3" name="joint_2" min="-1.57" max="1.57" step="0.005">
                    </div>
                </form>
                <button class="btn btn-default" onclick="$('#joint_3').val(0)">Reset 0 </button>
                <form>
                    <div class="form-group">
                        <label for="joint_3">Joint 3:</label>
                        <input type="range" id="joint_4" name="joint_3" min="-1.57" max="1.57" step="0.005">
                    </div>
                </form>
                <button class="btn btn-default" onclick="$('#joint_4').val(0)">Reset 0 </button>
                <form>
                    <div class="form-group">
                        <label for="joint_4">Joint 4:</label>
                        <input type="range" id="joint_5" name="joint_4" min="-1.57" max="1.57" step="0.005">
                    </div>
                </form>
                <button class="btn btn-default" onclick="$('#joint_5').val(0)">Reset 0 </button>
                <hr>
                
                <div class="row">
					    <div id="startArmButton" class="col-xs-12 col-ms-4 col-md-4 col-xl-4 text-center">
                            <button class="btn btn-default btn-lg" onclick="startArm()" style="background:lightgreen">Power arm</button>
                        </div>
                        <div class="col-xs-12 col-ms-4 col-md-4 col-xl-4 text-center">
                            <button class="btn btn-default btn-lg" onclick="getCurrentPosition()">Get current position</button>
                        </div>
                        <div class="col-xs-12 col-ms-4 col-md-4 col-xl-4 text-center">
                            <button class="btn btn-default btn-lg" onclick="sendPositionArm()">Send</button>
                        </div>
                    </div>
                    <hr>
                    <div class="row">
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12 text-center">
                            <button class="btn btn-default btn-lg" onclick="openModalToSavePosition()">Save position</button>
                        </div>
                    </div>
            </div>

        </div>
    </div>
    <hr>
    <div class="row">
        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
            <div class="container">
                <h2>Gripper</h2>
                <p>Open or Close the gripper</p>
                <div class="row">
                    <div class="col-xs-12 col-ms-6 col-md-6 col-xl-6 text-center">
                        <button class="btn btn-default btn-lg" onclick="openTheClipper()">Open</button>
                    </div>
                    <div class="col-xs-12 col-ms-6 col-md-6 col-xl-6 text-center">
                        <button class="btn btn-default btn-lg" onclick="closeTheClipper()">Close</button>
                    </div>
                </div>

            </div>
        </div>
    </div>
</div>

<!-- Modal -->
<div id="saveCurrentPositionModal" class="modal fade" role="dialog">
  <div class="modal-dialog">

    <!-- Modal content-->
    <div class="modal-content">
      <div class="modal-header">
        <button type="button" class="close" data-dismiss="modal">&times;</button>
        <h4 class="modal-title">Save position</h4>
      </div>
      <div class="modal-body">
        <p>Put a name the current position.</p>
        <div class="form-group">
            <label for="textIdInput">Name:</label>
            <input type="text" class="form-control" id="textIdInput">
        </div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-default btn-lg btn btn-default btn-lg-default" onclick="saveThePosition()">Save</button>
      </div>
    </div>

  </div>
</div>

<script>
    state_joint = "";
    
    function startArm(){
        torqueEnable = [];
        torqueEnable["torque_enable"] = true;
        
        callService('/joint_1/torque_enable', 'dynamixel_controllers/TorqueEnable', torqueEnable, function(val) {});
		
		callService('/joint_2/torque_enable', 'dynamixel_controllers/TorqueEnable', torqueEnable, function(val) {});
		
		callService('/joint_3/torque_enable', 'dynamixel_controllers/TorqueEnable', torqueEnable, function(val) {});
		
		callService('/joint_4/torque_enable', 'dynamixel_controllers/TorqueEnable', torqueEnable, function(val) {});
		
		callService('/joint_5/torque_enable', 'dynamixel_controllers/TorqueEnable', torqueEnable, function(val) {});
		
		callService('/gripper_revolute_joint/torque_enable', 'dynamixel_controllers/TorqueEnable', torqueEnable, function(val) {});
    }

    function sendPositionArm(){
        msgJoint = [];
        Joints = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5"];
        msgJoint["joints"] = Joints;
        Values = [parseFloat( $("#joint_1").val()),parseFloat( $("#joint_2").val()),parseFloat( $("#joint_3").val()),parseFloat( $("#joint_4").val()),parseFloat( $("#joint_5").val())];
        msgJoint["values"] = Values;
        msgJoint["time_plan"] = 10
        msgJoint["relative"] = false
        publish("/rt_traj_planner/commands/joint_by_joint", "robotnik_trajectory_planner/JointByJoint", msgJoint);
        console.log(msgJoint);


    }

    function getCurrentPosition(){
        updateCurrentPosition();
    }

    function openModalToSavePosition(){
        $("#textIdInput").val("");
        $("#saveCurrentPositionModal").modal('show');
    }

    function saveThePosition(){
        srvPoint = [];
        srvPoint["group_name"]="widowx_arm";
        srvPoint["id"] = $("#textIdInput").val();
        callService ('/rt_traj_manager/save_state', 'robotnik_trajectory_manager/SaveState', srvPoint, function (val){
            if(val.ret){
                $("#saveCurrentPositionModal").modal('hide');
                
            }else{
                alert("Error to save");
            }
        });
        
    }

    function openTheClipper(){
        msgFloat = [];
        msgFloat["data"] = 0;
        publish("/gripper_revolute_joint/command", "std_msgs/Float64", msgFloat);

    }


    function closeTheClipper(){
        msgFloat = [];
        msgFloat["data"] = 1.30;
        publish("/gripper_revolute_joint/command", "std_msgs/Float64", msgFloat);
        
    }

    function updateCurrentPosition(){
        if(state_joint == "")
            return false;
        aux_state = state_joint;
        aux_state["joint_names"].forEach(function(elem, index){
            if("joint_1"==elem){
                $("#joint_1").val(aux_state["actual"]["positions"][index]);
            }else if("joint_2"==elem){
                $("#joint_2").val(aux_state["actual"]["positions"][index]);
            }else if("joint_3"==elem){
                $("#joint_3").val(aux_state["actual"]["positions"][index]);
            }else if("joint_4"==elem){
                $("#joint_4").val(aux_state["actual"]["positions"][index]);
            }else if("joint_5"==elem){
                $("#joint_5").val(aux_state["actual"]["positions"][index]);
            }
        });

    }

    subscribe('/arm_controller/state', 'control_msgs/FollowJointTrajectoryFeedback', function(message) {
            state_joint = message;
    });    

    

    window.onload = function() {
        setTimeout(function () {
            updateCurrentPosition();
        }, 1000);
    };
        



</script>
