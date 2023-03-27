
<div class="" style="padding:15px; margin: 2%">
    <div class="row">
        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
            <div class="container">
                <h2><?php echo $lang['MISSION_TITLE'];?></h2>
                <p><?php echo $lang['MISSION_EXPLANATION'];?></p>
                
                <div class="panel panel-danger" id="divMissionFeedback">
					<div class="panel-heading">State Mission</div>
					<div class="panel-body">
						<p></p><label id="labelStateMission"></label><p>
						<p><label>Mission queue:</label></p>
						<p><label id="labelMissionQueue"></label><p>
						<div class="text-center">
							<button onclick="cancelAllMissions()" class="btn btn-danger btn-md"><span class="glyphicon glyphicon-stop"></span>Cancel all missions</button>
						</div>
					</div>
				</div>
				<p>
                    <button id="newMissionButton" type="button" class="btn btn-primary btn-sm" onclick="showMissionModal()">
                        <span class="glyphicon glyphicon-plus-sign"></span><?php echo $lang['MISSION_NEW_MISSION'];?>
                    </button>
                    <button id="deleteMissionButton" type="button" class="btn btn-warning btn-sm" onclick="deleteMission()" hidden>
                        <span class="glyphicon glyphicon-trash"></span><?php echo $lang['MISSION_DELETE_MISSION'];?>
                    </button>
                </p>
                <form>
                    <div class="form-group">
                        <label for="missionSelect"><?php echo $lang['MISSION_SELECT_A_MISSION'];?></label>
                        <select class="form-control" id="missionSelect">
                            <option><?php echo $lang['MISSION_CHOOSE_A_MISSION'];?></option>
                        </select>
                    </div>
                </form>
                <div id="missionPanelDiv" class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                    <div class=row>
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <p><label><?php echo $lang['MISSION_MISSION'];?>:</label><label id="nameMissionLabel"></label></p>
                            <p><label><?php echo $lang['MISSION_DESCRIPTION'];?>:<label><label id="descriptionMissionLabel"></label></p>
                        </div>
                        <div id=actionsDiv>
                            <!--div class="col-xs-12 col-ms-6 col-md-3 col-xl-3" id="action_1">
                                <div class="panel panel-success">
                                    <div class="panel-heading">
                                        <button type="button" class="btn btn-default btn-sm">
                                            <span class="glyphicon glyphicon-remove-circle"></span>
                                        </button>
                                        GoTo
                                    </div>
                                    <div class="panel-body">
                                        <label>Action: Init</label>
                                        <label>Description: </label>
                                    </div>
                                </div>
                            </div-->
                            
                        </div>
                        <div id="newAction" hidden>
                            <button type="button" class="btn btn-primary btn-lg" onclick="showActionSelect()" id="addActionButton" hidden>
                                <span class="glyphicon glyphicon-plus-sign"></span><?php echo $lang['MISSION_ADD_ACTION'];?>
                            </button>
                        </div>
                    </div>
                    <br>
                </div>
                <div id="saveNewMission" class="col-xs-12 col-ms-12 col-md-12 col-xl-12" hidden>
                    <div class=row>
                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                        <button type="button" class="btn btn-success btn-lg" onclick="saveNewMission()" id="saveNewMissionButton">
                            <?php echo $lang['MISSION_SAVE_MISSION'];?>
                        </button>
                    </div>
                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                        <button type="button" class="btn btn-warning btn-lg" onClick="window.location.reload();" id="cancelSaveNewMissionButton">
                            <?php echo $lang['MISSION_CANCEL'];?>
                        </button>
                    </div>
                    </div>
                </div>
                <div id="executeMission" class="col-xs-12 col-ms-12 col-md-12 col-xl-12" hidden>
                    <div class=row>
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            
                            <button type="button" class="btn btn-success btn-lg" onclick="executeTheMission()" id="executeTheMissionButton">
                                <?php echo $lang['MISSION_EXECUTE_MISSION'];?>
                            </button>
                        </div>
                    
                    </div>
                </div>


            </div>

        </div>
    </div>
</div>

<!-- Modal new mission-->
<div id="newMissionModal" class="modal fade" role="dialog">
  <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal">&times;</button>
                <h4 class="modal-title"><?php echo $lang['MISSION_MODAL_NEW_MISSION_TITLE'];?></h4>
            </div>
            <div class="modal-body">
                <div class="form-group">
                    <label for="newMissionName"><?php echo $lang['MISSION_MODAL_NAME_OF_MISSION'];?>:</label>
                    <input type="text" class="form-control" id="newMissionName">
                </div>
                <div class="form-group">
                    <label for="newMissionDescription"><?php echo $lang['MISSION_MODAL_DESCRIPTION_OF_MISSION'];?>:</label>
                    <textarea class="form-control" rows="5" id="newMissionDescription"></textarea>
                    
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-default" onclick = "newMissionMode()" data-dismiss="modal"><?php echo $lang['MISSION_MODAL_ADD_ACTIONS_IN_MISSION'];?></button>
            </div>
        </div>

  </div>
</div>


<!-- Modal add action-->
<div id="addActionModal" class="modal fade" role="dialog">
    <div class="modal-dialog">
        <div class="modal-content">
            <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal">&times;</button>
                <h4 class="modal-title"><?php echo $lang['MISSION_MODAL_ACTION_TITLE'];?></h4>
            </div>
            <div class="modal-body">
                <div class="form-group">
                    <label for="typeActionSelect"><?php echo $lang['MISSION_MODAL_ACTION_SELECT_TYPE_OF_ACTION'];?></label>
                    <select class="form-control" id="typeActionSelect">
                        <option value=-1><?php echo $lang['MISSION_MODAL_ACTION_CHOOSE_TYPE_OF_ACTION'];?></option>
                        <option value=0><?php echo $lang['MISSION_MODAL_ACTION_TYPE_GO_TO'];?></option>
                        <!--option value=4>GPS GoTo</option-->
                        <!--option value=1>Arm</option>
                        <option value=2>Gripper</option-->

                        <option value=5><?php echo $lang['MISSION_MODAL_ACTION_TYPE_DOCKING_TO_CHARGE'];?></option>
                        <option value=9><?php echo $lang['MISSION_MODAL_ACTION_TYPE_UNDOCKING'];?></option>
                        <!--option value=7><?php echo $lang['MISSION_MODAL_ACTION_TYPE_PICK'];?></option>
                        <option value=8><?php echo $lang['MISSION_MODAL_ACTION_TYPE_PLACE'];?></option-->
                        
                        
                    </select>
                </div>
                <div class="form-group" id="actionSelectDiv">
                    <label for="actionSelect"><?php echo $lang['MISSION_MODAL_ACTION_SELECT_AN_ACTION'];?></label>
                    <select class="form-control" id="actionSelect">
                        <option value=-1><?php echo $lang['MISSION_MODAL_ACTION_CHOOSE_AN_ACTION'];?></option>
                    </select>
                </div>
            </div>
            
            <div class="modal-footer">
                <button type="button" class="btn btn-default" onclick="addNewAction()"><?php echo $lang['MISSION_MODAL_ACTION_ADD_ACTION'];?></button>
            </div>
        </div>

    </div>
</div>

<script>


    var mode = 0;
    var id_mission = -1;
    var missions = [];
    var current_mission = [];
    actions = 0;
    typeActionSelected = -1;

    function changeMode(new_mode){
        mode=new_mode;
        actions = 0;
        switch(mode) {
            case 0:
                missionNotSelected();
                break;
            case 1:
                missionSelected();
                break;
            case 2:
                createANewMission();
                break;
            case 3:
                modifyMission();
                $("[name='errase_action']").show();
        }
    }
    

	subscribe('<?php echo $robot_dictionary["topic_robotnik_rms_state"]?>', 'robotnik_rms_msgs/status', function (data){
		$("#labelStateMission").text(data.message);
		var mission_queue = "";
		$.each(data.mission_queue, function( index, value ) {
			if(index > 0){
				mission_queue += "->";
			}
			mission_queue +=value;
		});	
		$("#labelMissionQueue").text(mission_queue);
		
	});
	
	function cancelAllMissions(){
		callService ("/robot/robotnik_rms/cancel_all_missions", "/robot/robotnik_rms", {}, function (value) {
			alertify.success(value.message);
		})
	}
	
    function missionNotSelected(){
        $("[name='action_group']").remove();
        $("#deleteMissionButton").hide();
        $("#executeMission").hide();
        $("#saveNewMission").hide();
        $("#addActionButton").hide();
        $('#newMissionButton').show();
        $("#missionPanelDiv").hide();
    }

    function missionSelected(){
        $("[name='action_group']").remove();
        if(missions.length == 0 ){
            return;
        }
        
        missions["missions"].forEach(function(item) {
            if(item["id_mission"]==id_mission){
                $('#nameMissionLabel').text(item["name_mission"]);
                $('#descriptionMissionLabel').text(item["description"]);
                
                item["actions"].forEach(function (action){
                    name_action = "";
                    if(action["type_action"]==0){
                        name_action = '<?php echo $lang['MISSION_MODAL_ACTION_TYPE_GO_TO'];?>';
                    }else if(action["type_action"]==1){
                        name_action = "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_ARM'];?>";
                    }else if(action["type_action"]==2){
                        name_action = "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_GRIPER'];?>";
                    }else if(action["type_action"]==3){
                        name_action = "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_QR'];?>";
                    }else if(action["type_action"]==4){
                        name_action = "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_GO_TO_GPS'];?>";
                    }else if(action["type_action"]==5){
					name_action= "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_DOCKING_TO_CHARGE'];?>";
					}
					else if(action["type_action"]==6){
					name_action= "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_APROACH_KAIROS'];?>";
					}
					else if(action["type_action"]==7){
					name_action= "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_PICK'];?>";
					}
					else if(action["type_action"]==8){
					name_action= "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_PLACE'];?>";
					}
                    else if(action["type_action"]==9){
					name_action= "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_UNDOCKING'];?>";
					}
                    else if(action["type_action"]==10){
					name_action= "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_CALL_ARM'];?>";
					}
					else{
                        name_action = "<?php echo $lang['MISSION_MODAL_ACTION_TYPE_UNWKNOW'];?>";
                    }
                    addNewActionCard(action["type_action"], name_action, action["name_action"], action["description"], action["id_action"], true)
                });
            }


        });
        $("#deleteMissionButton").show();
        $("#executeMission").show();
        $("#saveNewMission").hide();
        $("#addActionButton").hide();
        $('#newMissionButton').show();
        $("#newAction").hide();
        $("#missionPanelDiv").show();

    }

    function createANewMission(){
        actions=0;
        $("#deleteMissionButton").hide();
        $("#executeMission").hide();
        $("#saveNewMission").show();
        $("#addActionButton").show();
        $("[name='action_group']").remove();
        $('#newMissionButton').hide();
        $('#nameMissionLabel').text($('#newMissionName').val());
        $('#descriptionMissionLabel').text($('#newMissionDescription').val());
        $("#missionPanelDiv").show();
        
    }

    function newMissionMode(){
        changeMode(2);
        $("#newAction").show();

    }

    function showActionSelect(){

        $('#actionSelectDiv').hide();
        $('#typeActionSelect').val(-1);
        $('#addActionModal').modal('show');
    }

    function showMissionModal(){
        $('#newMissionName').val("");
        $('#newMissionDescription').val("");
        $('#newMissionModal').modal('show');
    }


    function addNewActionCard(type_action, type_action_name, name_action, description, id_action, hidde=false){
        hidden = ""
        if(hidde){
            hidden = 'style="display: none;"'
        }
        if(description==""){
            description = "Not description";
        }
        panel = '<div class="col-xs-12 col-ms-6 col-md-3 col-xl-3" id="action_'+actions+'" name="action_group" type_action="'+type_action+'" id_action="'+id_action+'" name_action="'+name_action+'" description_action="'+description+
        '"><div class="panel panel-success"><div class="panel-heading"><button type="button" onclick="deleteAction('+actions+
        ')"class="btn btn-default btn-sm" '+ hidden +' name="errase_action" ><span class="glyphicon glyphicon-remove-circle"></span></button>'+type_action_name+
        '</div><div class="panel-body"><p><label><?php echo $lang['MISSION_CUBE_ACTION_ACTION_NAME'];?>: '+name_action+'</label></p><p><label><?php echo $lang['MISSION_CUBE_ACTION_DESCRIPTION'];?>:'+description+ '</label></p><label name="id_action" hidden>'+id_action+'</label></div></div></div>'
        $('#actionsDiv').append(panel);

        actions = actions +1;
    }

    function deleteAction(id){
        elem = "#action_"+id;
        $(elem).remove();
    }


    function saveNewMission(){
        cards = $("[name=action_group]");
        mission = {};
        mission["name_mission"] = $('#nameMissionLabel').text();
        mission["description"] = $('#descriptionMissionLabel').text();
        mission["id_mission"] = 0;
        mission["actions"] = [];
        
        action_aux = {};
        jQuery.each(cards, function( i, val ) {
            action_aux["type_action"] = parseInt(val.attributes.type_action.nodeValue);
            action_aux["id_action"] = val.attributes.id_action.nodeValue;
            action_aux["description"] = val.attributes.description_action.nodeValue;
            action_aux["name_action"] = val.attributes.name_action.nodeValue;
            mission["actions"].push(Object.assign({}, action_aux));
        });
        srvMission = {};
        srvMission["mission"] = mission;
        console.log(srvMission);
        callService ('<?php echo $robot_dictionary["service_robotnik_rms_base"]?>/add_mission_in_list', 'robotnik_rms_msgs/missionManager', srvMission, function (val){
            if(val.success){
                alertify.success("<?php echo $lang['MISSION_ADDED_MISSION_SUCCESSFUL'];?>");
                window.location.reload();
            }else{
                alertify.success("<?php echo $lang['MISSION_ADDED_MISSION_ERROR'];?>: " + val.message);
            }
        });
    }


    function chargeMissions(){
        callService ('<?php echo $robot_dictionary["service_robotnik_rms_base"]?>/get_list_of_mission', '/robotnik_rms/get_list_of_mission', {}, function (val){
            missions = val;
            $('#missionSelect').empty();
            $('#missionSelect').append($('<option>', {
                value: -1,
                text: "<?php echo $lang['MISSION_CHOOSE_A_MISSION'];?>"
                }));
            $('[name=options]').val(-1);
            jQuery.each(missions["missions"], function( i, val ) {
                $('#missionSelect').append($('<option>', {
                    value: parseInt(val["id_mission"]),
                    text: val["name_mission"]
                }));
            });
            
        });
    }


function chargeListPoi (){
    callService ("<?php echo $robot_dictionary["service_poi_interactive_marker_base"]?>/read_pois", "poi_manager/UpdatePOIs", {}, function (value) {
        $("#actionSelect").empty();
        $('#actionSelect').append($('<option>', {
                value: -1,
                text: "<?php echo $lang['MISSION_MODAL_CHOOSE_A_POINT'];?>"
                }));
            $('#actionSelect').val(-1);
        $.each(value["pose_list"], function(index, val_each){
            $('#actionSelect').append($('<option>', {
                value: val_each["label"],
                text: val_each["label"]
            }));
        });
        
    });
}

function chargeListArm(){
    callService ("/rt_traj_manager/get_states", "/rt_traj_manager", {}, function (value) {
        $("#actionSelect").empty();
        $('#actionSelect').append($('<option>', {
                value: -1,
                text: "<?php echo $lang['MISSION_MODAL_CHOOSE_AN_ARM_POSITION'];?>"
                }));
            $('#actionSelect').val(-1);
        $.each(value["ret"], function(index, val_each){
            $('#actionSelect').append($('<option>', {
                value: val_each,
                text: val_each
            }));
        });
        
    });
}

function chargeListGripper(){
    $("#actionSelect").empty();
    $('#actionSelect').append($('<option>', {
        value: -1,
        text: "<?php echo $lang['MISSION_MODAL_CHOOSE_A_GRIPPER_ACTION'];?>"
    }));
    $('#actionSelect').val(-1);
    $('#actionSelect').append($('<option>', {
        value: "0",
        text: "<?php echo $lang['MISSION_MODAL_CHOOSE_A_GRIPPER_ACTION_OPEN'];?>"
    }));
    $('#actionSelect').append($('<option>', {
        value: "1",
        text: "<?php echo $lang['MISSION_MODAL_CHOOSE_A_GRIPPER_ACTION_CLOSE'];?>"
    }));
}


function chargeListGPSMissions(){
    callService ("/robotnik_gps_waypoints/get_routes", "robotnik_gps_waypoints_msgs/GetRoutes", {}, function (value) {
        $("#actionSelect").empty();
        $('#actionSelect').append($('<option>', {
                value: -1,
                text: "<?php echo $lang['MISSION_MODAL_CHOOSE_A_GPS_MISSION'];?>"
                }));
            $('#actionSelect').val(-1);
        $.each(value["routes"]["routes"], function(index, val_each){
            $('#actionSelect').append($('<option>', {
                value: val_each.id,
                text: val_each.name
            }));
        });
        
    });
}


function chargeListNotActionID(){
	text_action = ""
	$( "#typeActionSelect option:selected" ).each(function() {
            text_action = $( this ).text();
        });
    $("#actionSelect").empty();
    $('#actionSelect').append($('<option>', {
		value: 0,
		text: text_action
	}));
	$('#actionSelect').val(0);
}



function executeTheMission(){
    if(id_mission==-1){
        return;
    }
    srv= {};
    srv["id_mission"] = parseInt(id_mission);
    callService ("<?php echo $robot_dictionary["service_robotnik_rms_base"]?>/add_mission_in_queue", "robotnik_rms_msgs/idMission", srv, function (value) {
        if(value["success"]){
            alertify.success('<?php echo $lang['MISSION_ADD_MISSION_IN_QUEUE'];?>');
        }else{
            alertify.error('<?php echo $lang['MISSION_ADD_MISSION_IN_QUEUE_ERROR'];?>');
        }
    });
}

function deleteMission(){
    alertify.confirm("<?php echo $lang['MISSION_DELETE_MISSION_DIALOG'];?>",
    function(){
        if(id_mission==-1){
            return;
        }
        srv= {};
        srv["id_mission"] = parseInt(id_mission);
        callService ("<?php echo $robot_dictionary["service_robotnik_rms_base"]?>/delete_mission", "robotnik_rms_msgs/idMission", srv, function (value) {
            if(value["success"]){
                alertify.success('<?php echo $lang['MISSION_DELETE_MISSION_SUCCESSFUL'];?>');
                window.location.reload();
            }else{
                alertify.error('<?php echo $lang['MISSION_DELETE_MISSION_ERROR'];?>');
            }
        });
        
    },
    function(){
        return;
    }).setHeader('<?php echo $lang['MISSION_DELETE_MISSION_DIALOG'];?>');;
    
}



function addNewAction(){
    if($('#actionSelect').val()!=-1){
        type_action_text = "";
        type_action = -1;
        $( "#typeActionSelect option:selected" ).each(function() {
            type_action = $( this ).val();
            type_action_text = $( this ).text();
        });
        id_action = -1;
        text_action = "";
        $( "#actionSelect option:selected" ).each(function() {
            id_action = $( this ).val();
            text_action = $( this ).text();
        });


        addNewActionCard(type_action, type_action_text, text_action, "", id_action, false);
        $('#addActionModal').modal('hide');        
    }else{
        alertify.error("<?php echo $lang['MISSION_ADD_NEW_ACTION_CHOOSE_A_ACTION_ERROR'];?>");
    }
}


    $("#missionSelect" ).change(function () {
        $( "#missionSelect option:selected" ).each(function() {
            id_mission = $( this ).val();
        });
        if(id_mission == -1){
            changeMode(0);
        }else{
            changeMode(1);
        }
    }).change();

    $("#typeActionSelect" ).change(function () {
        $( "#typeActionSelect option:selected" ).each(function() {
            typeActionSelected = $( this ).val();
        });
        if(typeActionSelected == 0){
            chargeListPoi ();
            $('#actionSelectDiv').show();

        }else if(typeActionSelected == 1){
            chargeListArm ();
            $('#actionSelectDiv').show();

        }else if(typeActionSelected == 2){
            chargeListGripper ();
            $('#actionSelectDiv').show();
        }else if(typeActionSelected == 3){
            chargeListNotActionID ();
            $('#actionSelectDiv').show();
        }else if(typeActionSelected == 4){
            chargeListGPSMissions ();
            $('#actionSelectDiv').show();
	}else if(typeActionSelected == 5){
	    chargeListNotActionID ();
	    $('#actionSelectDiv').show();
        }else if(typeActionSelected == 6){
            chargeListNotActionID ();
            $('#actionSelectDiv').show();
        }else if(typeActionSelected == 7){
            chargeListNotActionID ();
            $('#actionSelectDiv').show();
        }else if(typeActionSelected == 8){
            chargeListNotActionID ();
            $('#actionSelectDiv').show();
        }else if(typeActionSelected == 9){
            chargeListNotActionID ();
            $('#actionSelectDiv').show();
        }else if(typeActionSelected == 10){
            chargeListNotActionID ();
            $('#actionSelectDiv').show();
        }
        else{
            $('#actionSelectDiv').hide();
        }
    }).change();

    $( document ).ready(function() {
        
        changeMode(0);
        chargeMissions();
        
    });
</script>


