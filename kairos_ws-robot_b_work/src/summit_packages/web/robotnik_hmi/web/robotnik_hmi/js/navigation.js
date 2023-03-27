//jquery init
$(document).ready(function() {
	
		$('div#occupancygrid').width('70%');/* $( "#urdf" ).width(); */
	
		$('div#panel').width('25%');
	
		var anchodiv =  $('div#tabs-1').width();
		
		var anchonav = $('div#occupancygrid').width();
	
		var altonav = anchonav / 1.33;
	
		var anchopanel = anchodiv - anchonav - 1550;
	
		$('div#tabs-1').height(altonav);
	
		$(window).resize(function(){
			location.reload();
			}
		);
	
		//$('div#panel').width(anchopanel);
	
	
		//$('div#tabs-1').width(ancho);
		
		// Create the main viewer.
		var viewer = new ROS3D.Viewer({
			  divID : 'occupancygrid',
			width : anchonav,
			height : altonav - 250,
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
		
		/*var interactiveMarkerClient = new ROS3D.InteractiveMarkerClient({
		  ros : ros,
		  camera : viewer.camera,
		  rootObject : viewer.selectableObjects,
		  tfClient : tfClient,
		  topic: namespace+'/map_nav_manager_interactive_marker'
		});*/
		
		var poiInteractiveMarkerClient = new ROS3D.InteractiveMarkerClient({
			ros : ros,
			camera : viewer.camera,
			rootObject : viewer.selectableObjects,
			tfClient : tfClient,
			topic: namespace+'/poi_interactive_marker'
		  });
		
	});
	
	
	// MAP SERVER
	function startMapServer(){
		if(mapping == true){
			window.alert("Error. Mapping has to be off before running a map server");
			return;
		}
	
		if(map_server==true){
	
			window.alert("Map server already running");
	
		}else{
	
			var file_name = $('#filename').val();
			var checkbox_default = $('#checkbox_map_default').is(":checked");
	
			if(file_name == ''){
				window.alert("startMapServer: loading default map");
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
	
			window.alert("startMapServer: request successfully sent.");
			
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
			window.alert("Stopping Map Server");
	
		}else{
			window.alert("Map Server not running");
	
		}
	}
	
	// LOCALIZATION
	function startLocalization(){
		if(mapping == true){
			window.alert("Error. Mapping has to be off before running localization node");
			return;
		}
	
		if(map_server == false){
			window.alert("Localization node needs a map server");
		}
		
		if(localization==true){
	
			window.alert("Localization already running");
	
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
	
			window.alert("startLocalization: request successfully sent.");
			
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
			window.alert("Stopping Localization node");
	
		}else{
			window.alert("Localization node is not running");
	
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
			window.alert("autorun disabled!");
		});	
	
		
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
			window.alert("autorun enabled!");
		});	
		
	}
	
	function goIndex(){
		window.location.href = "index.html";
	}
	
	function goMapping(){
		window.location.href = "mapping.html";
	}