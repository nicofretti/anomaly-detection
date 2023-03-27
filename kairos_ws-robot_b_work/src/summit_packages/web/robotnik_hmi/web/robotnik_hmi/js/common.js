var ros = new ROSLIB.Ros({
    url : 'ws://'+hostname+':9090'
});

var mapping = false
var navigation = false
var localization = false
var map_server = false


//reload the page automatically when it is resized
/*$(window).bind('resize', function(e)
{
  if (window.RT) clearTimeout(window.RT);
  window.RT = setTimeout(function()
  {
    this.location.reload(false); 
  }, 100);
});*/

//jquery init
$(document).ready(function() {
    
        $('div#occupancygrid').width('70%');/* $( "#urdf" ).width(); */
    
        $('div#panel').width('50%');
    
        var anchodiv =  $('div#tabs-1').width();
        
        var anchonav = $('div#occupancygrid').width();
    
        var altonav = anchonav / 1.33;
    
        var anchopanel = anchodiv - anchonav - 1550;
    
        $('div#tabs-1').height(altonav);
    
        $(window).resize(function(){
            location.reload();
            }
        );
    
        //$('div#occupancygrid').width(anchopanel);
    
        //$('div#tabs-1').width(ancho);
        
        // Create the main viewer.
        var viewer = new ROS3D.Viewer({
              divID : 'occupancygrid',
            width : anchonav,
            height : altonav - 150,
            antialias : true,
            background: '#002233',
            cameraPose : {x: 3, y: 3, z: 1}
        });
    
        // Add a grid.
        viewer.addObject(new ROS3D.Grid({
            color:'#0181c4',
              cellSize: 0.5,
              num_cells: 20
        }));
    
        // Setup a client to listen to TFs.
        var tfClient = new ROSLIB.TFClient({
          ros : ros,
          fixedFrame : map_frame,
          angularThres : 0.01,
          transThres : 0.01,
          rate : 10.0,
          serverName: namespace+'/tf2_web_republisher',
          repubServiceName: namespace+'/republish_tfs'
        });
    
        // Setup the URDF client.
        var urdfClient = new ROS3D.UrdfClient({
          ros : ros,
          param: namespace+'/robot_description',
          tfClient : tfClient,
          rootObject: viewer.scene
        });
        
        // Setup the marker client.
        var gridClient = new ROS3D.OccupancyGridClient({
          ros : ros,
          rootObject : viewer.scene,
          continuous: true,
          topic: map_topic
        });
        
        var poiInteractiveMarkerClient = new ROS3D.InteractiveMarkerClient({
			ros : ros,
			camera : viewer.camera,
			rootObject : viewer.selectableObjects,
			tfClient : tfClient,
			topic: namespace+'/poi_interactive_marker'
		  });
        
        //readPOIs();

        autorun_function();
        
        //nav-bar scroll configuration
        (function($) {
            "use strict"; // Start of use strict
          
            // Smooth scrolling using jQuery easing
            $('a.js-scroll-trigger[href*="#"]:not([href="#"])').click(function() {
              if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') && location.hostname == this.hostname) {
                var target = $(this.hash);
                target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
                if (target.length) {
                  $('html, body').animate({
                    scrollTop: (target.offset().top - 44)
                  }, 750, "easeInOutExpo");
                  return false;
                }
              }
              alert( "algo");
            });
          
            // Closes responsive menu when a scroll trigger link is clicked
            $('.js-scroll-trigger').click(function() {
              $('.navbar-collapse').collapse('hide');
            });
          
            // Activate scrollspy to add active class to navbar items on scroll
            $('body').scrollspy({
              target: '#mainnavbar',
              offset: 44
            });
          
          })(jQuery); // End of use strict
        
    });

function autorun_function(){
    // assign function to onclick property of checkbox
    var aut=document.getElementById("autorun_default");
    aut.onclick = function() {
        // access properties using this keyword
        if ( this.checked ) {
            // if checked ...
            //alert( this.value );
            enableLocalizationAutorun();
        } else {
            // if not checked ...
            disableLocalizationAutorun();
        }        
    };
    
}

function startMapping(){
	
	if(map_server == true){
        window.alert("Error. map_server has to be switched off before running a mapping");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error. map_server has to be switched off before running a mapping";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        return;
	}
	if(mapping == true){
        window.alert("Error: mapping already running");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error: mapping already running";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }else{
		var svc = new ROSLIB.Service({  
			ros : ros,
			name : namespace+'/map_nav_manager/start_mapping',
			messageType : 'std_srv/Trigger'
		});

		svc.callService(function(res){
			console.log("Response received");
            });	
            document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info: Mapping process initiated.";
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            //window.alert("Mapping process initiated. Press OK.");	            
	}
}

function stopMapping(){

	if(mapping == false){
        window.alert("Error: mapping process not initiated");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error: mapping process not initiated";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }else{
		var svc = new ROSLIB.Service({  
			ros : ros,
			name :  namespace+'/map_nav_manager/stop_mapping',
			messageType : 'std_srv/Trigger'
		});

		svc.callService(function(res){
			console.log("stopMapping: Response received");
		});
		//window.alert("stopMapping:Mapping process stopped. Press OK.");
		document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info: Mapping process stopped.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }
}

function saveMap(){

	if(mapping==false){

		window.alert("Error. SLAM-Gmapping node not initiated");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error. SLAM-Gmapping node is not initiated.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
	}else{

		var file_name = $('#filename').val();
		var checkbox_default = $('#checkbox_map_default').is(":checked");

		if(file_name == ''){
            window.alert("Error. The name of the map can not be empty.");
            document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error. The name of the map can not be empty.";
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        }else{

			console.log("map name = %s, save as default = %d",file_name, checkbox_default);
			
			var svc = new ROSLIB.Service({  
				ros : ros,
				name :  namespace+'/map_nav_manager/save_map',
				messageType : 'map_nav_manager/SetFilename'
			});

			var data = new ROSLIB.ServiceRequest({
				name : file_name,
				use_it_by_default: checkbox_default
			});

			svc.callService(data,function(res){
				console.log("Response received");
			}); 

            //window.alert("El mapa se ha guardado con éxito.");
            document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info: Map saved successfully.";
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        }
	}

}

// MAP SERVER
function startMapServer(){
    if(mapping == true){
        window.alert("Error. Mapping has to be off before running a map server");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error: Mapping has to be off before running a map server";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        return;
    }

    if(map_server==true){

        window.alert("Map server already running");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error: Map server already running";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }else{

        var file_name = $('#filename').val();
        var checkbox_default = $('#checkbox_map_default').is(":checked");

        if(file_name == ''){
            //window.alert("startMapServer: loading default map");
            document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info: loading default map.";
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        }

        console.log(file_name);
        
        var svc = new ROSLIB.Service({  
            ros : ros,
            name : namespace + '/map_nav_manager/start_map_server',
            messageType : 'map_nav_manager/SetFilename'
        });

        var data = new ROSLIB.ServiceRequest({
            name : file_name,
            use_it_by_default: checkbox_default
        });

        svc.callService(data,function(res){
            console.log("startMapServer: Response received");
        });	

        //window.alert("startMapServer: request successfully sent.");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info: Map server initiated.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }

}
	
function stopMapServer(){
    if(map_server==true){
        var svc = new ROSLIB.Service({  
                ros : ros,
                name : namespace + '/map_nav_manager/stop_map_server',
                messageType : 'std_srv/Trigger'
        });

        var data = new ROSLIB.ServiceRequest({
            
        });
        svc.callService(data,function(res){
            console.log("stopMapServer: Response received");
        });	
        //window.alert("Stopping Map Server");

    }else{
        window.alert("Map Server not running");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info: Map Server stopped.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }
}

// LOCALIZATION
function startLocalization(){
    if(mapping == true){
        window.alert("Error. Mapping has to be off before running localization node");        
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error. Mapping has to be off before running localization node.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        return;
    }

    if(map_server == false){
        window.alert("Error. Localization node needs a map server node running");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error. Localization node needs a map server.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        return;
    }
    
    if(localization==true){

        //window.alert("Localization already running");

    }else{
    
        var svc = new ROSLIB.Service({  
            ros : ros,
            name : namespace + '/map_nav_manager/start_localization',
            messageType : 'std_srv/Trigger'
        });

        var data = new ROSLIB.ServiceRequest({
        });

        svc.callService(data,function(res){
            console.log("startLocalization: Response received");
        });	

        //window.alert("startLocalization: request successfully sent.");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info. Localization node initiated.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }

}

function stopLocalization(){
    
    if(localization==true){
        
        var svc = new ROSLIB.Service({  
                ros : ros,
                name : namespace + '/map_nav_manager/stop_localization',
                messageType : 'std_srv/Trigger'
        });

        var data = new ROSLIB.ServiceRequest({
            
        });
        svc.callService(data,function(res){
            console.log("stopLocalization: Response received");
        });	
        //window.alert("Stopping Localization node");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info. Localization node stopped.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }else{
        //window.alert("Localization node is not running");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info. Localization node is not running.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }
}


// Disables the autorun of all the loc & nav components
function disableLocalizationAutorun(){
    
    
    var svc = new ROSLIB.Service({  
        ros : ros,
        name : namespace + '/map_nav_manager/autorun_loc_and_nav',
        messageType : 'std_srv/SetBool'
    });

    var data = new ROSLIB.ServiceRequest({
        data: false
    });

    svc.callService(data,function(res){
        console.log("disableLocalizationAutorun: Response received");
        //window.alert("autorun disabled!");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info. Autorun disabled.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    });	

    
}


//NAVIGATION
function startNavigation(){

   if(navigation == true){
       window.alert("Error: navigation already running");
       document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error. Navigation already running.";
       document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }else{
   var svc = new ROSLIB.Service({  
       ros : ros,
       name : namespace+'/map_nav_manager/start_navigation',
       messageType : 'std_srv/Trigger'
   });

   svc.callService(function(res){
       console.log("Response received");
   });	
   //window.alert("Navigation process initiated. Press OK.");
   document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info. Navigation process initiated.";
   document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }

}

function stopNavigation(){

   if(navigation == false){
       window.alert("Error: navigation not running");
       document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Error. Navigation process is NOT running.";
       document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }else{
   var svc = new ROSLIB.Service({  
       ros : ros,
       name :  namespace+'/map_nav_manager/stop_navigation',
       messageType : 'std_srv/Trigger'
   });

   svc.callService(function(res){
       console.log("Response received");
   });
   //window.alert("Navigation process stopped. press OK.");
   document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info. Navigation process stopped.";
   document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;  
    }
}

// Enables the autorun of all the loc & nav components
function enableLocalizationAutorun(){
    var svc = new ROSLIB.Service({  
        ros : ros,
        name : namespace + '/map_nav_manager/autorun_loc_and_nav',
        messageType : 'std_srv/SetBool'
    });

    var data = new ROSLIB.ServiceRequest({
        data: true
    });

    svc.callService(data,function(res){
        console.log("enableLocalizationAutorun: Response received");
        //window.alert("autorun enabled!");
        document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info. Autorun enabled.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    });	
    
}

// Enables the autorun of all the loc & nav components
function rlcResumeMission(){
	var svc = new ROSLIB.Service({  
		ros : ros,
		name : '/robot_local_control/resume',
		messageType : 'std_srv/Trigger'
	});

	svc.callService(function(res){
		console.log("rlcResumeMission: Response received");
		//window.alert("autorun enabled!");
		document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info. Mission resumed";
		document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
	});	
	
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
