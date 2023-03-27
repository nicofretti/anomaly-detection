var ros = new ROSLIB.Ros({
    url : 'ws://'+hostname+':9090'
});

var mapping = false
var navigation = false
var localization = false
var map_server = false


function startNavigation(){

   if(navigation == true){
       window.alert("Error: navigation already running");
   }else{
   var svc = new ROSLIB.Service({  
       ros : ros,
       name : namespace+'/map_nav_manager/start_navigation',
       messageType : 'std_srv/Trigger'
   });

   svc.callService(function(res){
       console.log("Response received");
   });	
   window.alert("Navigation process initiated. Press OK.");
   
   }

}

function stopNavigation(){

   if(navigation == false){
       window.alert("Error: navigation not running");
   }else{
   var svc = new ROSLIB.Service({  
       ros : ros,
       name :  namespace+'/map_nav_manager/stop_navigation',
       messageType : 'std_srv/Trigger'
   });

   svc.callService(function(res){
       console.log("Response received");
   });
   window.alert("Navigation process stopped. press OK.");
   
   }
}

// TOPIC SUBSCRIBERS
var map_nav_state_sub = new ROSLIB.Topic({
   ros : ros,
   name : namespace + '/map_nav_manager/state',
   messageType : 'map_nav_manager/State'
});
   
// Topic handlers
map_nav_state_sub.subscribe(function(message) {
   mapping = message.mapping;
   navigation = message.navigation;
   localization = message.localization;
   map_server = message.map_server;
   
   if(mapping){
       document.getElementById("mapping_status").style.color = "Green";
       
       try{
           document.getElementById("button_save_map").disabled = false;
           document.getElementById("button_start_mapping").disabled = true;
           document.getElementById("button_stop_mapping").disabled = false;
       }catch(e){
           
       }
   }else{
       document.getElementById("mapping_status").style.color = "Red";
       try{
           document.getElementById("button_save_map").disabled = true;
           document.getElementById("button_start_mapping").disabled = false;
           document.getElementById("button_stop_mapping").disabled = true;
       }catch(e){
           
       }
   }
   if(navigation){
       document.getElementById("navigation_status").style.color = "Green";
       document.getElementById("button_start_navigation").disabled = true;
       document.getElementById("button_stop_navigation").disabled = false;
   }else{
       document.getElementById("navigation_status").style.color = "Red";
       document.getElementById("button_start_navigation").disabled = false;
       document.getElementById("button_stop_navigation").disabled = true;
   }
   if(localization){
       document.getElementById("localization_status").style.color = "Green";
       try{
           document.getElementById("button_start_localization").disabled = true;
           document.getElementById("button_stop_localization").disabled = false;
       }catch(e){
           
       }
       
   }else{
       document.getElementById("localization_status").style.color = "Red";
       try{
           document.getElementById("button_start_localization").disabled = false;
           document.getElementById("button_stop_localization").disabled = true;
       }catch(e){
           
       }
       
   }
   if(map_server){
       document.getElementById("map_server_status").style.color = "Green";
       try{
           document.getElementById("button_start_map_server").disabled = true;
           document.getElementById("button_stop_map_server").disabled = false;
       }catch(e){
           
       }
       
   }else{
       document.getElementById("map_server_status").style.color = "Red";
       try{
           document.getElementById("button_start_map_server").disabled = false;
           document.getElementById("button_stop_map_server").disabled = true;
       }catch(e){
           
       }
   }
   
}); 