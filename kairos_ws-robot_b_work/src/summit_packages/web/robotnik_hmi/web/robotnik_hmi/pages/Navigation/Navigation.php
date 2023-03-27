
<div class="" style="padding:15px; margin: 2%">
    <div class="row">
        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
            <div class="container">
                <div class="row">
                <div id="divAutonomousState">
                    <h1><?php echo $lang['NAVIGATION_TITLE'] ?></h1>
                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                        <p><?php echo $lang['NAVIGATION_EXPLANATION'] ?></p>
                        <div class="row">
                            <div class="col-xs-7 col-ms-7 col-md-7 col-xl-7">
                                <div class="row">
                                    <div class="form-group col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                        <form>
                                            <label for="selectPoi"><?php echo $lang['NAVIGATION_LIST_SELECT'] ?>
                                                <p></p>
                                                <button type="button" class="btn btn-default btn-sm" onclick="chargeListPoi()" id="buttonRefreshList">
                                                    <span class="glyphicon glyphicon-refresh"></span>
                                                </button>
                                                <button type="button" class="btn btn-default btn-sm" onclick="PanelAddPoint()" id="buttonAddPoint">
                                                    <span class="glyphicon glyphicon-plus"></span>
                                                </button>
                                            </label>

                                            <select class="form-control" id="selectPoi">
                                                
                                            </select>
                                        </form>
                                    </div>
                                </div>
                                <div class="row center-block">
                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                        <button type="button" class="btn btn-danger btn-sm center-block" id="buttonStop" onclick="Stop()"><?php echo $lang['NAVIGATION_STOP'] ?></button>
                                    </div>
                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                        <button type="button" class="btn btn-primary btn-sm center-block" id="buttonGoTo" onclick="GoTo()"><?php echo $lang['NAVIGATION_GO_TO_POINT'] ?></button>
                                    </div>
                                </div>
                              
                                <div class="row center-block">
                                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12 text-center">
                                        <label id="labelMsgToRobot" hidden="true"><?php echo $lang['NAVIGATION_MESSAGE_TO_ROBOT']?></label><label id = "labelmsg" hidden="true"></label>
                                    </div>
                                </div>
                            </div>
                            <div class="col-xs-5 col-ms-5 col-md-5 col-xl-5">
                                <div class="panel panel-default" id="divPanelPoint" hidden="true">
                                    <div class="panel-heading"><?php echo $lang['NAVIGATION_POINT_TITLE']?></div>
                                    <div class="panel-body">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <div class="row">
                                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <label><?php echo $lang['NAVIGATION_POINT_LABEL']?></label>
                                                    </div>
                                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <input type="text" class="form-control" id="inputLabel">
                                                    </div>
                                                    
                                                </div>
                                                <hr>
                                                <div class="row">
                                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <label><?php echo $lang['NAVIGATION_POINT_X']?></label>
                                                    </div>
                                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <input type="number" class="form-control" id="inputX">
                                                    </div>
                                                </div>
                                                <hr>
                                                <div class="row">
                                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <label><?php echo $lang['NAVIGATION_POINT_Y']?></label>
                                                    </div>
                                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <input type="number" class="form-control" id="inputY">
                                                    </div>
                                                </div>
                                                <hr>
                                                <div class="row">
                                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <label><?php echo $lang['NAVIGATION_POINT_THETA']?></label>
                                                    </div>
                                                    <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                        <input type="number" class="form-control" id="inputTheta">
                                                    </div>
                                                </div>
                                                <div class="row" id="divGetPoint" hidden="true">
                                                    <hr>
                                                    <button type="button" class="btn btn-warning btn-sm center-block" onclick="GetRobotPosition()"><?php echo $lang['NAVIGATION_POINT_GET_ROBOT_POSITION']?></button>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id = "divAddPoint" hidden="true">
                                                <hr>
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                                    <button type="button" class="btn btn-primary btn-sm" onclick="AddPoint()"><?php echo $lang['NAVIGATION_ADD_POINT']?></button>
                                                </div>
                                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6 text-right">
                                                    <button type="button" class="btn btn-danger btn-sm" onclick="CancelPoint()"><?php echo $lang['NAVIGATION_CANCEL_POINT']?></button>
                                                </div>
                                            </div>
                                        </div>
                                        
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="panel-body" id="occupancygrid"></div>
                        </div>
                        
                    </div>
                </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>

function GoTo(){
    GoToPoint($('#selectPoi').val());
}

function AddPoint(){
    if($("#inputLabel").val() != "" && $("#inputX").val()!= "" && $("#inputY").val()!= "" && $("#inputTheta").val()!= ""){
        addPoint($("#inputLabel").val(), parseFloat($("#inputX").val()), parseFloat($("#inputY").val()), parseFloat($("#inputTheta").val()));
            $("#divPanelPoint").hide();
            $("#divGetPoint").hide();
            $("#divAddPoint").hide();
            $("#inputLabel").attr("disabled", true);
            $("#inputLabel").val("");
            $("#inputX").attr("disabled", true);
            $("#inputX").val("");
            $("#inputY").attr("disabled", true);
            $("#inputY").val("");
            $("#inputTheta").val("");
            $("#selectPoi").attr("disabled", false);
            $("#buttonGoTo").attr("disabled", false);
            $("#buttonStop").attr("disabled", false);
            $("#buttonAddPoint").attr("disabled", false);
            $("#buttonRefreshList").attr("disabled", false);
        }else{
            
        }
    }
    
    function RefreshPointsInMap (){
         RefreshPoisMap ();
    }
    
    function CancelPoint(){
        $("#divPanelPoint").hide();
        $("#divGetPoint").hide();
        $("#divAddPoint").hide();
        $("#inputLabel").attr("disabled", true);
        $("#inputLabel").val("");
        $("#inputX").attr("disabled", true);
        $("#inputX").val("");
        $("#inputY").attr("disabled", true);
        $("#inputY").val("");
        $("#selectPoi").attr("disabled", false);
        $("#buttonGoTo").attr("disabled", false);
        $("#buttonStop").attr("disabled", false);
        $("#buttonAddPoint").attr("disabled", false);
        $("#buttonRefreshList").attr("disabled", false);
        var optionSelected = $("#selectPoi option:selected");
        if(optionSelected.length == 1){
            $("#selectPoi").val($("#selectPoi option:first").val());
            $("#labelmsg").hide();
            $("#labelMsgToRobot").hide();
            
            $("#divPanelPoint").show();
            $("#inputLabel").attr("disabled", true);
            $("#inputLabel").val(optionSelected.val());
            $("#inputX").attr("disabled", true);
            $("#inputX").val(optionSelected.attr("meta-x"));
            $("#inputY").attr("disabled", true);
            $("#inputY").val(optionSelected.attr("meta-y"));
            $("#inputTheta").attr("disabled", true);
            $("#inputTheta").val(optionSelected.attr("meta-theta"));
        }
    }
    
    function PanelAddPoint(){
        $("#divPanelPoint").show();
        $("#divGetPoint").show();
        $("#divAddPoint").show();
        $("#inputLabel").attr("disabled", false);
        $("#inputLabel").val("");
        $("#inputX").attr("disabled", false);
        $("#inputX").val("");
        $("#inputY").attr("disabled", false);
        $("#inputY").val("");
        $("#inputTheta").attr("disabled", false);
        $("#inputTheta").val("");
        $("#selectPoi").attr("disabled", true);
        $("#buttonGoTo").attr("disabled", true);
        $("#buttonStop").attr("disabled", true);
        $("#buttonAddPoint").attr("disabled", true);
        $("#buttonRefreshList").attr("disabled", true);
        $("#selectPoi").val($("#selectPoi option:first").val());
    };
    
    $( "#selectPoi" ).change(function() {
        $("#labelmsg").hide();
        $("#labelMsgToRobot").hide();
        var optionSelected = $("option:selected", this);
        $("#divPanelPoint").show();
        $("#inputLabel").attr("disabled", true);
        $("#inputLabel").val(optionSelected.val());
        $("#inputX").attr("disabled", true);
        $("#inputX").val(optionSelected.attr("meta-x"));
        $("#inputY").attr("disabled", true);
        $("#inputY").val(optionSelected.attr("meta-y"));
        $("#inputTheta").attr("disabled", true);
        $("#inputTheta").val(optionSelected.attr("meta-theta"));
    });
    
    function GetRobotPosition (){
        var pos = getPosition();
        $("#inputX").val(pos["x"].toFixed(4));
        $("#inputY").val(pos["y"].toFixed(4));
        $("#inputTheta").val(pos["theta"].toFixed(4));
        
    }
    
    /* 
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

function addPoint (label, x, y, theta){
    var srv = getStructUpdatePOIs ();
    srv.pose_list[0].label = label;
    srv.pose_list[0].pose.x = x;
    srv.pose_list[0].pose.y = y;
    srv.pose_list[0].pose.theta = theta;
    callService ("<?php echo $robot_dictionary["service_poi_interactive_marker_add_point"];?>", "poi_manager/UpdatePOIs", srv, function (value){
        chargeListPoi ();
        alertify.succes('<?php echo $lang['NAVIGATION_ADDED']?>');
    });
}

function chargeListPoi (){
    callService ("<?php echo $robot_dictionary["service_poi_manager_read_points"];?>", "poi_manager/UpdatePOIs", {}, function (value) {
        $("#selectPoi").html("");
        var emptyList = true;
        $.each(value["pose_list"], function(index, val_each){
            if (val_each["label"] != undefined){
                var optionAdd = "<option value="+val_each["label"]+" meta-x="+val_each["pose"]["x"]+" meta-y="+val_each["pose"]["x"]+" meta-theta="+val_each["pose"]["theta"]+" >"+val_each["label"]+"</option>";
                $("#selectPoi").append(optionAdd);
                emptyList = false;
            }
        });
        if(!emptyList){
            $("#selectPoi").val($("#selectPoi option:first").val());
            $("#labelmsg").hide();
            $("#labelMsgToRobot").hide();
            var optionSelected = $("#selectPoi option:selected");
            $("#divPanelPoint").show();
            $("#inputLabel").attr("disabled", true);
            $("#inputLabel").val(optionSelected.val());
            $("#inputX").attr("disabled", true);
            $("#inputX").val(optionSelected.attr("meta-x"));
            $("#inputY").attr("disabled", true);
            $("#inputY").val(optionSelected.attr("meta-y"));
            $("#inputTheta").attr("disabled", true);
            $("#inputTheta").val(optionSelected.attr("meta-theta"));
        }
    });
}

function GoToPoint (label){
    
    
    var srv = getStructGoTo ();
    srv["name"] = label;
    callService ("<?php echo $robot_dictionary["service_poi_interactive_marker_goto_tag"];?>", "map_nav_manager/GoToTag", srv ,function(value){
        $("#labelmsg").show();
        $("#labelMsgToRobot").show();
        if (value.ret){
            $("#labelmsg").text(value.msg + ", go to " + label);
            alertify.succes('<?php echo $lang['NAVIGATION_GO_TO_TAG'];?>: '+ label);
        }else{
            $("#labelmsg").text("Error ->" + value.msg);
            alertify.error('<?php echo $lang['NAVIGATION_GO_TO_TAG_ERROR'];?>: '+ label);
        }
    });

}

function Stop(){
    var srv = getStructBoolAcction();
    srv.data = true;
    callService("<?php echo $robot_dictionary["service_poi_interactive_marker_stop_goto"];?>", "std_srvs/SetBool", srv, function(value){
        $("#labelmsg").show();
        $("#labelMsgToRobot").show();
        $("#labelmsg").text("Stop");
        alertify.error('<?php echo $lang['NAVIGATION_STOP'];?>');
        
    });
}

function RefreshPoisMap (){
    var srv = getStructBoolAcction();
    srv.data = true;
    callService("<?php echo $robot_dictionary["service_poi_interactive_marker_load_points"];?>", "std_srvs/SetBool", srv, function(value){
        
    })
}


var posX;
var posY;
var posTheta;

function getPosition(){
    var position = {};
    position["x"] = posX;
    position["y"] = posY;
    position["theta"] = posTheta;
    return position;
}


subscribe('<?php echo $robot_dictionary["topic_robot_pose"];?>', 'geometry_msgs/PoseWithCovarianceStamped', function(message) {
    posX = message.pose.pose.position.x;
    posY = message.pose.pose.position.y;
    // yaw (z-axis rotation).z + message.orientation.x * message.orientation.y);
    var siny = +2.0 * (message.pose.pose.orientation.w * message.pose.pose.orientation.z);
    var cosy = +1.0 - 2.0 * (message.pose.pose.orientation.z * message.pose.pose.orientation.z);
    posTheta =Math.atan2(siny, cosy);
});

var mapping;
var navigation;
var localization;
var map_server;
var AutonomousReady;
var FollowReady;
var DockingReady;
var DockingQRReady;

subscribe('<?php echo $robot_dictionary["topic_map_nav_state"];?>','map_nav_manager/State', function(message) {
   mapping = message.mapping;
   navigation = message.navigation;
   localization = message.localization;
   map_server = message.map_server;
   
   if(mapping){
       $("#liAAutonomous").attr("href", "#divAutonomousError");
       $("#liAFollow").attr("href", "#divFollowError");
       $("#liADocking").attr("href", "#divDowckingError");
       $("#liADockingQR").attr("href", "#divDowckingQRError");
       AutonomousReady = false;
       FollowReady = false;
       DockingReady = false;
       DockingQRReady = false;
   }else{
       $("#liAFollow").attr("href", "#divFollowError");
       $("#liADocking").attr("href", "#divDockingState");
       $("#liADockingQR").attr("href", "#divDockingQrState");
       
       FollowReady = true;
       DockingReady = true;
       DockingQRReady = true;
       if (navigation && localization && map_server){
           
           $("#liAAutonomous").attr("href", "#divAutonomousState");
           AutonomousReady = true;
       }else{
           $("#liAAutonomous").attr("href", "#divAutonomousError");
           AutonomousReady = false;
       }
       //AutonomousReady = true;
       //$("#liAAutonomous").attr("href", "#divAutonomousState");
   }
   
}); 


function getStructLabeledPose (){
    return {
        label: "",
        pose: {
            x : 0,
            y : 0,
            theta : 0
            
        }
    };
}

function getStructUpdatePOIs (){
    return {
        pose_list : [
                getStructLabeledPose()
        ]
    }
}

function getStructGoTo (){
    return {
        name : ""
    };
}

function getStructBoolAcction(){
    return {
        data : false
    };
}


global_frame = '<?php echo $robot_dictionary["global_frame"]?>';

$(document).ready(function(){
    RefreshPoisMap ();
    chargeListPoi();
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
        addRobot('<?php echo $robot_dictionary["description"] ?>');
        setMap(global_frame);
        interactiveMarkerClient('<?php echo $robot_dictionary["topic_poi_interactive_marker"] ?>');
        setMap('<?php echo $robot_dictionary["topic_map"] ?>');
});

</script>
