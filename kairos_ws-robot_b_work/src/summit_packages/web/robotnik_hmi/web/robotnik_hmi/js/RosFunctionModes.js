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
    callService ("/poi_interactive_marker/add_pois_2d", "poi_manager/UpdatePOIs", srv, function (value){
        chargeListPoi ();
        
    });
}

function chargeListPoi (){
    callService ("/poi_manager/read_pois", "poi_manager/UpdatePOIs", {}, function (value) {
        $("#selectPoi").html("");
        var emptyList = true;
        $.each(value["pose_list"], function(index, val_each){
            if (val_each["label"] != undefined){
                var optionAdd = "<option value="+val_each["label"]+" meta-x="+val_each["pose"]["x"]+" meta-y="+val_each["pose"]["x"]+" meta-theta="+val_each["pose"]["x"]+" >"+val_each["label"]+"</option>";
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
    callService ("/poi_interactive_marker/goto_tag", "map_nav_manager/GoToTag", srv ,function(value){
        $("#labelmsg").show();
        $("#labelMsgToRobot").show();
        if (value.ret){
            $("#labelmsg").text(value.msg + ", go to " + label);
        }else{
            $("#labelmsg").text("Error ->" + value.msg);
        }
    });

}

function Stop (){
    var srv = getStructBoolAcction();
    srv.data = true;
    callService("/poi_interactive_marker/stop_goto", "std_srvs/SetBool", srv, function(value){
        $("#labelmsg").show();
        $("#labelMsgToRobot").show();
        $("#labelmsg").text("Stop");
    });
}

function RefreshPoisMap (){
    var srv = getStructBoolAcction();
    srv.data = true;
    callService("/poi_interactive_marker/load_poi", "std_srvs/SetBool", srv, function(value){
        
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


subscribe('/robot_pose', 'geometry_msgs/Pose', function(message) {
    posX = message.position.x;
    posY = message.position.y;
    // yaw (z-axis rotation).z + message.orientation.x * message.orientation.y);
    var siny = +2.0 * (message.orientation.w * message.orientation.z);
    var cosy = +1.0 - 2.0 * (message.orientation.z * message.orientation.z);
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

subscribe('/map_nav_manager/state','map_nav_manager/State', function(message) {
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

var docking_time=0;
activeDocking = false;
window.setInterval(function (){
    docking_time++;
    if(docking_time>2){
	docking_time=2;
        $("#labelDockingMsg").text("");
        $("#buttonActiveDocking").prop("disabled", false);
        $("#buttonDisableDocking").prop("disabled", true);
    }else{
	$("#buttonActiveDocking").prop("disabled", activeDocking);
	$("#buttonDisableDocking").prop("disabled", !activeDocking);
    }
}, 500);


subscribe('/auto_docking_log', 'std_msgs/String', function(message){
    docking_time=0;
    activeDocking=true;
    $("#labelDockingMsg").text(message.data);
});




var followMeState;

subscribe('/turtlebot_follower/state_follow', 'std_msgs/Bool', function(message){
    $("#liAFollow").attr("href", "#divFollowState");
    $('#buttonActiveFollow').removeClass('active');
    $('#buttonActiveFollow').removeClass('disabled');
    $('#buttonDisableFollow').removeClass('active');
    $('#buttonDisableFollow').removeClass('disabled');
    
    if(message["data"]){
        $('#buttonActiveFollow').addClass('disabled');
        $('#buttonDisableFollow').addClass('active');
    }else{
        $('#buttonActiveFollow').addClass('active');
        $('#buttonDisableFollow').addClass('disabled');
    }
    followMeState = message["data"];
});

function changeStateFollow (active = false){
    var srv = {'state': 0};
    if (active){
        srv["state"] = 1;
    }
    callService("/turtlebot_follower/change_state", "turtlebot_msgs/SetFollowState", srv, function(value){
        if(value["result"]!=0){
            $("#liAFollow").attr("href", "#divFollowError");
        }
    });
}

function changeStateDocking (active = false){
    var srv={};
    activeDocking=active;
    if (active){
	callService("/auto_docking_service","std_srvs/Empty",srv, function(){});
	
    }else{
	callService("/cancel_auto_docking","std_srvs/Empty",srv, function(){});
    }
}

var activeDockingQr = false;

function changeDockingQR (active = false){
    var srv={};
    srv["data"]=active;
    activeDockingQr = active;
    if(active){
        callService("/poi_interactive_marker/docking_qr","std_srvs/SetBool" ,srv, function(){});
    }else{
        callService("/poi_interactive_marker/docking_qr","std_srvs/SetBool" ,srv, function(){});
    }
}

subscribe('/docker/status', 'actionlib_msgs/GoalStatusArray', function(val){
    if(val.status_list.length>0){
	if(val.status_list[val.status_list.length-1].status != 1){
		activeDockingQr = false;
	}else{
		activeDockingQr = true;
	}
    }else{
	activeDockingQr = false;

    }
    $('#buttonActiveDockingQr').removeClass('active');
    $('#buttonActiveDockingQr').removeClass('disabled');
    $('#buttonDisableDockingQr').removeClass('active');
    $('#buttonDisableDockingQr').removeClass('disabled');
    if(activeDockingQr){

        $('#buttonActiveDockingQr').addClass('disabled');
        $('#buttonDisableDockingQr').addClass('active');
	if(val.status_list.length>0){
		$('#labelDockingQrMsg').text(val.status_list[val.status_list.length-1].text);
	}else{
		$('#labelDockingQrMsg').text("Disabled");
	}
         
    }else{
        $('#buttonActiveDockingQr').addClass('active');
        $('#buttonDisableDockingQr').addClass('disabled');
	if(val.status_list.length>0){
		$('#labelDockingQrMsg').text(val.status_list[val.status_list.length-1].text);
	}else{
		$('#labelDockingQrMsg').text("Disabled empty status");
	}
        
    }   
});

