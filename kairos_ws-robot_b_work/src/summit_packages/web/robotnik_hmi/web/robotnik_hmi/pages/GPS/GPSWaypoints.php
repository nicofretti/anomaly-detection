<div class="backgroundPageWeb" style="margin-top:60px">
    <div class='container'>
        <div class="row">
            <div class="panel panel-danger">
                <div class="panel-heading">
                    <label>GPS</label>
                </div>
                <div class="panel-body">
                    <div class="row">
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <div id="divMap" style="height: 500px;width: 100%"></div>
                        </div>
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <div class="row" id="divModifyPointManually">
                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                    <div class="text-center">
                                        <button type="button" class="btn btn-default btn-md" onclick="modifyPositionPointManually()">
                                            <span class="glyphicon glyphicon-map-marker"></span> Modify point position
                                        </button>
                                    </div>
                                </div>
                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                    <div class="text-center">
                                        <button type="button" class="btn btn-default btn-md" onclick="modifyRotationPointManually()">
                                            <span class="glyphicon glyphicon-map-marker"></span> Modify point orientation
                                        </button>
                                    </div>
                                </div>
                            </div>
                            <div class="row" id="divSaveDiscardPointManually">
                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                    <div class="text-center">
                                        <button type="button" class="btn btn-default btn-md" onclick="savePointManually()">
                                            <span class="glyphicon glyphicon-map-marker"></span> Save point
                                        </button>
                                    </div>
                                </div>
                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                    <div class="text-center">
                                        <button type="button" class="btn btn-default btn-md" onclick="DiscardPointManually()">
                                            <span class="glyphicon glyphicon-map-marker"></span> Discard point
                                        </button>
                                    </div>
                                </div>
                            </div>
                            <div class="row" id="divManagerPoints">
                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                    <div class="text-center">
                                        <button type="button" class="btn btn-default btn-md" onclick="addPointManually()">
                                            <span class="glyphicon glyphicon-map-marker"></span> Add point manually
                                        </button>
                                    </div>
                                </div>
                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                    <div class="text-center">
                                        <button type="button" class="btn btn-default btn-md" onclick="addRobotPoint()">
                                            <span class="glyphicon glyphicon-map-marker"></span> Add robot point
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div class="row">
            <div class="panel panel-danger">
                <div class="panel-heading">
                    <label>Routes</label>
                </div>
                <div class="panel-body">
                <div class="panel panel-success" id="divExecutionActions" hidden>
                    <div class="panel-heading">
                        <label>Running route information</label>
                    </div>
                    <div class="panel-body">
                        <div class="row">
                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                <div class="row">
                                    <div class="col-xs-10 col-ms-9 col-md-12 col-xl-12">
                                        <label id="labelRMSStateMessage">Info</label>
                                    </div>
                                    <div class="col-xs-2 col-ms-3 col-md-12 col-xl-12">
                                        <div class="text-right">
                                            <button class="btn btn-warning btn-lg"  onclick="cancelRoute()">Cancel route</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                    <div id="divChosseRoute" class="row">
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <form>
                                <div class="form-group">
                                    <label for="listRoutes">Select a route:</label>
                                    <button type="button" class="btn btn-default" id="buttonAddRoute" onclick="buttonNewRoute()">
                                        <span class="glyphicon glyphicon-plus"></span>Add new route
                                    </button>
                                    <select class="form-control" id="listRoutes">
                                        <option value="-1">--Choose a route--</option>
                                    </select>
                                </div>
                            </form>
                        </div>
                    </div>
                    
                    <div id="divManagerRoute" class="row">
                        <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                            <button type="button" class="btn btn-default btn-sm" id="buttonDeleleteRoute" onclick="buttonDeleleteRoute(id_route)">
                                <span class="glyphicon glyphicon-trash"></span> Delete
                            </button>
                        </div>
                        <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6 text-right">
                            <button type="button" class="btn btn-default btn-sm" id="buttonModifyRoute" onclick="buttonModifyRoute(id_route)">
                                <span class="glyphicon glyphicon-pencil"></span> Modify
                            </button>
                        </div>
                        <hr>
                    </div>
                    <div id="divInformationRoute" class="row">
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <div class="form-group">
                                <label for="nameRoute" class="control-label">Name route:</label>
                                <input class="form-control" id="nameRoute" type="text">
                            </div>
                        </div>
                        
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <div class="form-group">
                                <label for="routeDescription" class="control-label">Route description:</label>
                                <textarea class="form-control" id="routeDescription" type="text" rows="5"  maxlength="500"></textarea>
                                </div>
                            </div>
                        </div>
                        <hr>
                        
                        <div id="divTablePoints" class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <table class="display responsive nowrap" id="tablePointsInRoute"  cellspacing="0" width="100%">
                                <thead>
                                    <tr>
                                        <!--th>
                                            <label>id</label>
                                        </th-->
                                        <th>
                                            <label>Position</label>
                                        </th>
                                        
                                        <th>
                                            <label>Longitude</label>
                                        </th>
                                        <th>
                                            <label>Latitude</label>
                                        </th>
                                        <th>
                                            <label>Orientation</label>
                                        </th>
                                        <th>
					<!--th>
                                            <label>Navigation type</lable>
                                        </th-->
                                        <th>
                                            <label>Local planner</lable>
                                        </th>

                                        </th>
                                    </tr>
                                </thead>
                            </table>
                        </div>
                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <div class="row" id="divSaveCancel">
                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                    <div class="text-center">
                                        <button class="btn btn-success btn-lg" onclick="saveTheRoute()">Save</button>
                                    </div>
                                </div>
                                <div class="col-xs-6 col-ms-6 col-md-6 col-xl-6">
                                    <div class="text-center">
                                        <button class="btn btn-danger btn-lg"  onclick="notSaveRoute()">Cancel</button>
                                    </div>
                                </div>
                            </div>
                            
                        </div>
                        <div class="row" id="divStartRoute">
                            <div class="text-center">
                                <button class="btn btn-success btn-lg" onclick="startRoute()">Start route</button>
                            </div> 
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

</div>


<script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyCVM-OKAfzF4UcexNum8mV39ujIJAFoWpk"></script>


<script>

var mode_page = 0;
var points = [];
var path = [];
var gps_longitude = 0.0;
var gps_latitude = 0.0;
var gps_altitude = 0.0;
var gps_orientation = 0.0;
var first_msg_gps = false;
var map;
var robot_marker;
var uluru;
var id_route = -1;
var point_manual_mode = false;
var step_point_manual = -1;
var array_point_route = [];
var aux_arrow_point_route = null;

var gpsLongitude = -0.4619901;
var gpsLatitude = 39.5079743;
var gpsAltitude = 1.00;
var turnGPS = 0;
var table_points_in_route;


var select_local_planne = `
    <select id="selectNavigationType" class="form-control">
    <option value="eband" selected="selected">Eband</option>
    <option value="teb">Teb</option>
    <option value="pure_pursuit">Pure pursuit</option>
    <option value="pure_pursuit_aw">Pure pursuit aw</option>
    </select>
    `;

var select_global_planne = `
    <div>
    <select id="selectNavigationGlobalType" class="form-control">
        <option value="SBPLLatticePlanner" selected="selected">SBPL</option>
        <option value="GlobalPlanner">GlobalPlanner</option>
        <option value="none">None</option>
    </select>
    <div class="form-check">
        <input type="checkbox" class="form-check-input" id="inputOnlyFirstGlobalPlanner">
        <label class="form-check-label" for="inputOnlyFirstGlobalPlanner">Only the first point</label>
    </div>
    </div>
    `;

$( document ).ready(function() {
    table_points_in_route = $('#tablePointsInRoute').DataTable({
        "bPaginate": false,
        "bFilter": false,
        "ordering": false,
        "columns": [
            //{"data": "id"},
            {"data": "position"},
            {"data": "longitude"},
            {"data": "latitude"},
            {"data": "orientation"},
	        {"data": "local_planner"},
            //{"data": null, "className": "center", "defaultContent": select_local_planne},

            {"data": null, "className": "center", "defaultContent": '<button name="deleteAPoint" type="button" class="btn btn-default btn-sm"><span class="glyphicon glyphicon-remove-circle"></span></button>'}
	    
        ]
    });

    $('#tablePointsInRoute tbody').on('click', 'td [name="deleteAPoint"]', function (e) {
        console.log("heyyyyy deleted point" + $(this).closest('tr').prevAll().length);
        deleteAPoint($(this).closest('tr').prevAll().length);
    } );
    initMap();
    getRoutes();
});

function changeModePage(mode){
    mode_page = mode;
    if(mode==0){ //Not selected a route
        point_manual_mode = false; 
        array_route_arrow = [];
        id_route = -1;
        step_point_manual = -1;
        clearAllPoints();
        $("#divManagerPoints").hide();
        $("#divModifyPointManually").hide();
        $("#divSaveDiscardPointManually").hide();
        $("#divTablePoints").hide();
        $("#divManagerRoute").hide();
        $("#buttonAddRoute").show();
        $("#divInformationRoute").hide();
        $("#divSaveCancel").hide();
        $("#divStartRoute").hide();
        $("#nameRoute").prop('disabled', true);
        $("#routeDescription").prop('disabled', true);
        $("button[name='deleteAPoint']").hide();
        
            
        if(table_points_in_route != undefined){
            table_points_in_route.clear().draw();
        }
        
    }else if(mode==1){ //Selected a route
        if(id_route==-1){
            changeModePage(0);
            return;
        }
        point_manual_mode = false;
        array_route_arrow = [];
        step_point_manual = -1;
        clearAllPoints();
        $("#divManagerPoints").hide();
        $("#divModifyPointManually").hide();
        $("#divSaveDiscardPointManually").hide();
        $("#divTablePoints").show();
        chargePointsRoute(id_route);
        $.each($("#listRoutes option:selected"), function(){
            $("#nameRoute").val($(this).prop("text"));
            $("#routeDescription").val($(this).attr("description"));
        });
        $("#divManagerRoute").show();
        $("#buttonAddRoute").show();
        $("#divInformationRoute").show();
        $("#divSaveCancel").hide();

        $("#divStartRoute").show();
        $("#nameRoute").prop('disabled', true);
        $("#routeDescription").prop('disabled', true);
        $("button[name='deleteAPoint']").hide();
    }else if(mode==2){ //Modify a route
        chargePointsRoute(id_route);
        step_point_manual = -1;
        $("#divTablePoints").show();
        $("#divManagerPoints").show();
        $("#divManagerRoute").hide();
        $("#buttonAddRoute").hide();
        $("#divSaveCancel").show();
        $("#divStartRoute").hide();
        $("#nameRoute").prop('disabled', false);
        $("#routeDescription").prop('disabled', false);
        $("button[name='deleteAPoint']").show();
        
    }else if(mode==3){
        array_route_arrow = [];
        step_point_manual = -1;
        clearAllPoints();
        step_point_manual = -1;
        if(table_points_in_route != undefined){
            table_points_in_route.clear().draw();
        }
        $("#nameRoute").val("");
        $("#routeDescription").val("");
        $("#divInformationRoute").show();
        $("#divTablePoints").show();
        $("#divManagerPoints").show();
        $("#divManagerRoute").hide();
        $("#buttonAddRoute").hide();
        $("#divSaveCancel").show();
        $("#divStartRoute").hide();
        $("#nameRoute").prop('disabled', false);
        $("#routeDescription").prop('disabled', false);
        $("button[name='deleteAPoint']").show();
    }
}

$("#listRoutes" ).change(function () {
  $( "#listRoutes option:selected").each(function() {
    id_route = parseInt($( this ).val());
    if(id_route == -1){
      changeModePage(0);
    }else{
      changeModePage(1);
    }
  });
}).change();




var markerRobotIcon = {
  path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
  scale: 5,
  fillColor: 'red',
  strokeColor: 'red'
};

function initMap() {
    if(!first_msg_gps){
        uluru = {lat: 39.5079743, lng: -0.4619901};
    }
    if(gps_longitude != 0.0 && gps_latitude != 0.0){
        uluru = {lat: gps_longitude, lng: gps_latitude};
    }

    map = new google.maps.Map(document.getElementById('divMap'), {
        zoom: 50,
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

    marker = new google.maps.Polyline({
      path: [uluru, uluru],
      icons: [{
        icon: markerRobotIcon,
        offset: '100%'
      }],
      map: map
    });
    
    map.addListener('click', function(e) {
        clickInMap(e);
    });
    
    map.addListener('mousemove', function(e){
        mouseMoveInMap(e);
    });

}

function placeMarkerAndPanTo(latLng, map) {
  var marker = new google.maps.Marker({
    position: latLng,
    map: map
  });
  map.panTo(latLng);
}




function refreshMap() {
    marker.setPath([{lat: gpsLatitude, lng: gpsLongitude}, {lat: gpsLatitude+Math.sin(turnGPS)*0.00001, lng: gpsLongitude+Math.cos(turnGPS)*0.00001}]);
}


function topicPointGPS (message){
    gpsLongitude = message["longitude"];
    gpsLatitude = message["latitude"];
    gpsAltitude = message["altitude"];
    uluru = {lat: gpsLatitude, lng: gpsLongitude};
    if(!first_msg_gps){
        first_msg_gps = true;
        initMap();
    }
    refreshMap();
}
subscribe("<?php echo $robot_dictionary["topic_fix"]?>","sensor_msgs/NavSatFix", topicPointGPS);

var turnGPS;

function gpsOdometryFilter (message){
  var v = message.pose;
  var quaternion ={
    x: v.pose.orientation.x,
    y: v.pose.orientation.y,
    z: v.pose.orientation.z,
    w: v.pose.orientation.w
  };
  var turn = QuaternionToEulerAngle(quaternion);

  turnGPS =  (turn.yaw);

}
subscribe("<?php echo $robot_dictionary["gpsOdometryFilterTopic"]?>","nav_msgs/Odometry", gpsOdometryFilter);



function addRobotPoint(){
    var data_add_in_table = {};
    data_add_in_table["id"] = -1;
    data_add_in_table["position"] = parseInt(table_points_in_route.rows().count());
    data_add_in_table["longitude"] = parseFloat(gpsLongitude).toFixed(10);
    data_add_in_table["latitude"] = parseFloat(gpsLatitude).toFixed(10);
    data_add_in_table["orientation"] =  parseFloat(turnGPS).toFixed(10);
    alertify.alert("Choose a local planner." + select_local_planne,
    function(){
        data_add_in_table["local_planner"] = $("#selectNavigationType").val();
        table_points_in_route.row.add(data_add_in_table).draw().node();
        updateMapFromTable();
        $("button[name='deleteAPoint']").show();
    }).setHeader('Local planner') ;
}


function addPointManually(){
    if(aux_arrow_point_route != null){
        aux_arrow_point_route.setMap(null);
    }
    aux_arrow_point_route = null;
    point_manual_mode = true;
    step_point_manual = 0;
    $("#divManagerPoints").hide();
    $("#divSaveDiscardPointManually").show();
    
}


var arrow_rotation;

function clickInMap(e){
    if(step_point_manual == 0){ //Add new point in map
        aux_arrow_point_route = addArrowInMap(e.latLng.lat(), e.latLng.lng(), 0, "#2DA229");
        arrow_rotation = 0;
        step_point_manual++;
    }else if(step_point_manual == 1){ //Change rotation
        var new_arrow_path = aux_arrow_point_route.getPath().i;
        var rotation = Math.atan2(e.latLng.lat() - new_arrow_path[0].lat() , e.latLng.lng() - new_arrow_path[0].lng());
        
        var new_long = new_arrow_path[0].lng() + Math.cos(rotation) * 0.00005;
        var new_lat = new_arrow_path[0].lat() + Math.sin(rotation) * 0.00005;
        var path = [
            {lat:new_arrow_path[0].lat(), lng:new_arrow_path[0].lng()},
            {lat:new_lat, lng:new_long}
        ];
        aux_arrow_point_route = changeArrowInMap(aux_arrow_point_route, path);
        arrow_rotation = rotation;
        $("#divModifyPointManually").show();
        step_point_manual++;
        
    }else {
        
    }
}

function mouseMoveInMap(e){
    if(step_point_manual == 1){
        test = aux_arrow_point_route.getPath();
        var new_arrow_path = aux_arrow_point_route.getPath().i;
        var rotation = Math.atan2(e.latLng.lat() - new_arrow_path[0].lat() , e.latLng.lng() - new_arrow_path[0].lng());
        
        var new_long = new_arrow_path[0].lng() + Math.cos(rotation) * 0.00005;
        var new_lat = new_arrow_path[0].lat() + Math.sin(rotation) * 0.00005;
        var path = [
            {lat:new_arrow_path[0].lat(), lng:new_arrow_path[0].lng()},
            {lat:new_lat, lng:new_long}
        ];
        aux_arrow_point_route = changeArrowInMap(aux_arrow_point_route, path);
    }
}


function addArrowInMap(latitude, longitude, orientation, color="#000000"){
    var lineSymbol = {
        path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW
    };
    var latitude2 = latitude + Math.sin(orientation)*0.00005;
    var longitude2 = longitude + Math.cos(orientation)*0.00005;
    var mark = new google.maps.Polyline({
        path: [{lat: latitude, lng: longitude}, {lat: latitude2, lng: longitude2}],
            strokeColor: color,
            icons: [{
                icon: lineSymbol,
                offset: '100%'
        }],
        map: map
    });
    return mark;
}


function addPathInMap(latitude1, longitude1, latitude2, longitude2, color="#00FF00"){
    
    var mark = new google.maps.Polyline({
        path: [{lat: latitude1, lng: longitude1}, {lat: latitude2, lng: longitude2}],
        geodesic: true,
        strokeColor: color,
        strokeOpacity: 1.0,
        strokeWeight: 2
    });

    mark.setMap(map);

    return mark;
}

function changeArrowInMap(arrow, path){
    arrow.setMap(null);
    arrow.setPath(path);
    arrow.setMap(map);
    return arrow;
}


function savePointManually(){
    if(aux_arrow_point_route == null){
        alertify.alert("Not added point","You need add the point in the map, before saving the point. Use the map to add the point.", function(){
        
        });
        return;
    }
    var data_add_in_table = {};
    data_add_in_table["id"] = -1;
    data_add_in_table["position"] = table_points_in_route.rows().count();
    data_add_in_table["longitude"] = aux_arrow_point_route.getPath().i[0].lng();
    data_add_in_table["latitude"] = aux_arrow_point_route.getPath().i[0].lat();
    data_add_in_table["orientation"] =  arrow_rotation;
    
    alertify.alert("Choose a local planner." + select_local_planne,
    function(){
        data_add_in_table["local_planner"] = $("#selectNavigationType").val();
        table_points_in_route.row.add(data_add_in_table).draw().node();
        $("button[name='deleteAPoint']").show();
        if(aux_arrow_point_route != null){
            aux_arrow_point_route.setMap(null);
        }
        DiscardPointManually();
    }).setHeader('Local planner');
    
}


function DiscardPointManually(){
    $("#divManagerPoints").show();
    $("#divSaveDiscardPointManually").hide();
    $("#divModifyPointManually").hide();
    if(aux_arrow_point_route != null){
        aux_arrow_point_route.setMap(null);
    }
    aux_arrow_point_route = null;
    step_point_manual = -1;
    updateMapFromTable();
}

function modifyRotationPointManually(){
    step_point_manual = 1;
}

function modifyPositionPointManually(){
    DiscardPointManually();
    addPointManually();
    step_point_manual = 0;
}

function clearAllPoints(){
    points.forEach(function (data, index){
        data.setMap(null);
    });
    path.forEach(function (data, index){
        data.setMap(null);
    });
    points = [];
    path = [];
    
}

function updateMapFromTable(){
    clearAllPoints();
    var aux;
    table_points_in_route.rows().data().each(function (value, index){
        var latitude = parseFloat(value["latitude"]);
        var longitude = parseFloat(value["longitude"]);
        var orientation = parseFloat(value["orientation"]);
        points.push(addArrowInMap(latitude, longitude, orientation));
        if(index>=1){
            path.push(addPathInMap(latitude, longitude, aux["latitude"], aux["longitude"]));
        }
        aux = value;
    });
}



function deleteAPoint(position){
    var find = false;
    var aux_update_table = [];
    var index_point_to_delete = null;
    table_points_in_route.rows().data().each(function (value, index){
        if(value["position"].toString()==position.toString()){
            find = true;
            index_point_to_delete = index;
        }
        else if(find){
            aux = {};
            aux["latitude"] = value["latitude"];
            aux["longitude"] = value["longitude"];  
            aux["orientation"] = value["orientation"];  
            aux["local_planner"] = value["local_planner"];
            aux["position"] = (value["position"]-1);
            aux_update_table.push(aux);
            
        }else{
            aux = {};
            aux["latitude"] = value["latitude"];
            aux["longitude"] = value["longitude"];  
            aux["orientation"] = value["orientation"];  
            aux["local_planner"] = value["local_planner"];
            aux["position"] = (value["position"]);
            aux_update_table.push(aux);
            
        }
    });
    if(index_point_to_delete!=null){
        table_points_in_route.clear().draw();
        table_points_in_route.rows.add(aux_update_table).draw();
    }
    updateMapFromTable();
}

function getRoutes(){
    $('#listRoutes').empty();
    $('#listRoutes').append($('<option>', {
        value: -1,
        text: "--Chose a route--"
    }));
    callService ('<?php echo $robot_dictionary["service_robotnik_waypoints_manager_namespace"]?>/get_routes', 'robotnik_waypoints_manager_msgs/GetRoutes', {}, function (val){
        if(val["success"]){
            val["routes"]["routes"].forEach(function (item, index){
                if(item["goal_target_type"]=="GPS"){
                    $('#listRoutes').append($('<option>', {
                        value: item["id"],
                        text: item["name"],
                        description: item["description"]
                    }));
                }
            });
        }else{
            alertify.error(val["message"]);
        }
    });
    id_route = -1;
    changeModePage(0);
}

function startRoute(){
    if(id_route==-1){
        return;
    }
    var global_planner = [];
    alertify.alert("Choose a global planner." + select_global_planne,
    function(){
        global_planner.push($("#selectNavigationGlobalType").val());
        if($('#inputOnlyFirstGlobalPlanner').is(':checked')){
            leng_of_table = parseInt(table_points_in_route.rows().count());
            for(cont=1; cont < leng_of_table; cont++){
                global_planner.push("none");
            }
        }
        var srv = {
        id_route: id_route,
        controller: [],
        planner: global_planner,
        navigation_type: 'POINT_TO_POINT',
        goal_tolerance: {
            position: {x: 0.0, y: 0.0, z: 0.0},
            orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}
        }
        
    }
    console.log(srv);
    callService ('<?php echo $robot_dictionary["service_robotnik_waypoints_exe_namespace"]?>/set_goal_route', 'robotnik_waypoints_msgs/GoalIdRoute', srv, function (val){
        if(val["success"]){
            alertify.success("Executing route");
        }else{
            alertify.error("Error to try execute the route: " + val["message"]);
        }
    });
    }).setHeader('Global planner') ;

    
}

function cbWaypointsExecution(message){
    if(message.current_state == "SEND PLAN" || message.current_state == "NAVIGATING"){
        $("#labelRMSStateMessage").text(message.message);
        $("#divExecutionActions").show();
    }else{
        $("#labelRMSStateMessage").text("");
        $("#divExecutionActions").hide();
    }

}

subscribe('<?php echo $robot_dictionary["service_robotnik_waypoints_exe_namespace"]?>/rms_status','robotnik_waypoints_msgs/RMSState', cbWaypointsExecution);


function cancelRoute(){
    callService ('<?php echo $robot_dictionary["service_robotnik_waypoints_exe_namespace"]?>/cancel_goal', 'std_srvs/Trigger', {}, function (val){
        if(val["success"]){
            alertify.success(val["message"]);
        }else{
            alertify.error("Error: " + val["message"]);
        }
    });
}

function chargePointsRoute(id){
    table_points_in_route.clear().draw();
    srv = {"id_route": parseInt(id)}
    callService('<?php echo $robot_dictionary["service_robotnik_waypoints_manager_namespace"]?>/get_points_in_route', 'robotnik_waypoints_manager_msgs/GetPoints', srv, function(val){
        if(val["success"]){
            val["points"].forEach(function (item, index){
                var data_add_in_table = {};
                
                data_add_in_table["position"] = item["position_index"];
                data_add_in_table["longitude"] = item["y"];
                data_add_in_table["latitude"] = item["x"];
                data_add_in_table["orientation"] =  item["orientation"];
                data_add_in_table["local_planner"] = item["local_planner"];
                table_points_in_route.row.add(data_add_in_table).draw().node();
            })
            updateMapFromTable();
            if(mode_page==0 || mode_page ==1){
                $("button[name='deleteAPoint']").hide();
            }else{
                $("button[name='deleteAPoint']").show();
            }
        }else{
            alertify.error(val["message"]);
        }
    });
}

function buttonDeleleteRoute (id){
    if(id==-1){
        changeModePage(0);
        return;
    }
    alertify.confirm("Delete route","Do you want delete the route?",
    function(){
        srv = {"id": parseInt(id),
                "route": {
                    "id" : 0,
                    "name": "",
                    "description": "",
                    "points": []
                }
        }
        callService('<?php echo $robot_dictionary["service_robotnik_waypoints_manager_namespace"]?>/delete_routes', 'robotnik_waypoints_manager_msgs/RouteAction', srv, function(val){
            if(val["success"]){
                alertify.success('Ok');
                getRoutes();
            }else{
                alertify.error(val["message"]);
            }
        });
    },
    function(){
        alertify.error('Cancel');
    });
}

function buttonModifyRoute(id){
    if(id==-1){
        changeModePage(0);
        return;
    }else{
        changeModePage(2);
        $("button[name='deleteAPoint']").show();
            
    }
}




function saveTheRoute(){
    if($("#nameRoute").val()==""){
        alertify.error("It is neccesary a route name");
        return;
    }
    if(mode_page == 2){
        alertify.confirm("Update the route","Do you want update the route?",
        function(){
            srv_route = {"id": parseInt(id_route),
                "route": {
                    "id" : parseInt(id_route),
                    "name": $("#nameRoute").val(),
                    "description": $("#routeDescription").val(),
                    "goal_target_type": "GPS",
                    "points": []
                }
            }
            callService('<?php echo $robot_dictionary["service_robotnik_waypoints_manager_namespace"]?>/modify_route', 'robotnik_waypoints_manager_msgs/RouteAction', srv_route, function(val){
                if(val["success"]){
                    callService('<?php echo $robot_dictionary["service_robotnik_waypoints_manager_namespace"]?>/delete_all_point_in_route', 'robotnik_waypoints_manager_msgs/RouteAction', srv_route, function(val2){
                        if(val2["success"]){
                            table_points_in_route.rows().data().each(function (elem, index){
                                srv_points = {
                                    "fk_id": parseInt(id_route),
                                    "id": 0,
                                    "point": {}
                                }
                                aux = {
                                    id: 0,
                                    fk_id_route: parseInt(id_route),
                                    position_index: parseInt(elem["position"]),
                                    name: "",
                                    description: "",
                                    x: parseFloat(elem["latitude"]),
                                    y: parseFloat(elem["longitude"]),
                                    z: 0,
                                    frame: "robot_gps_link",
                                    orientation: parseFloat(elem["orientation"]),
                                    local_planner: elem["local_planner"]
            
                                }
                                srv_points["point"] = aux;
                                console.log("New points: ");
                                console.log(srv_points["point"]);
                                callService("<?php echo $robot_dictionary["service_robotnik_waypoints_manager_namespace"]?>/add_point_in_route", "robotnik_waypoints_manager_msgs/PointAction", srv_points, function(val3){
                                    if(!val3["success"]){
                                        //alertify.error("When update the point: " + val3["message"]);
                                    }else{
                                        
                                    }
                                });
                                
                            });

                            alertify.success('Update');
                            getRoutes();
                            changeModePage(0);
                        }else{
                            alertify.error("When clear the points: " + val2["message"]);
                        }
                    });
                    
                }else{
                    alertify.error("When update route: " + val["message"]);
                }
            });
        },
        function(){

        });
    }else if(mode_page==3){
        srv_route = {
            "id": parseInt(id_route),
            "route": {
                "id" : parseInt(id_route),
                "name": $("#nameRoute").val(),
                "description": $("#routeDescription").val(),
                "goal_target_type": "GPS",
                "points": []
            }
        }
        table_points_in_route.rows().data().each(function (elem, index){
            aux = {
                id: 0,
                fk_id_route: parseInt(id_route),
                position_index: parseInt(elem["position"]),
                name: "",
                description: "",
                x: parseFloat(elem["latitude"]),
                y: parseFloat(elem["longitude"]),
                z: 0,
                frame: "robot_gps_fix",
                orientation: parseFloat(elem["orientation"]),
                local_planner: elem["local_planner"]
            };
            srv_route["route"]["points"].push(Object.assign({}, aux));
        });
        console.log(srv_route["route"]);
        callService("<?php echo $robot_dictionary["service_robotnik_waypoints_manager_namespace"]?>/add_new_route", "robotnik_waypoints_manager_msgs/RouteAction", srv_route, function(val){
            if(val["success"]){
                alertify.success("Added");

                getRoutes();
                changeModePage(0);
            }else{
                alertify.error(val["message"]);
            }
        });
    }else{
        alertify.error("You cannot pass");
    }

}

function notSaveRoute(){
    if(mode_page == 2){
        changeModePage(1);
    }else{
        $('#listRoutes').val(-1);
        changeModePage(0);
    }
}

function buttonNewRoute(){
    $('#listRoutes').val(-1);
    changeModePage(3);
}




function EulerToQuaternion( ypr_values) // yaw (Z), pitch (Y), roll (X)
{

    // Abbreviations for the various angular functions
    var cy = Math.cos(ypr_values.yaw * 0.5);
    var sy = Math.sin(ypr_values.yaw * 0.5);
    var cp = Math.cos(ypr_values.pitch * 0.5);
    var sp = Math.sin(ypr_values.pitch * 0.5);
    var cr = Math.cos(ypr_values.roll * 0.5);
    var sr = Math.sin(ypr_values.roll * 0.5);

    var quaternions = {};
    quaternions.w = cy * cp * cr + sy * sp * sr;
    quaternions.x = cy * cp * sr - sy * sp * cr;
    quaternions.y = sy * cp * sr + cy * sp * cr;
    quaternions.z = sy * cp * cr - cy * sp * sr;
    return quaternions;
}



function normalizeRad (value){
    while (value > Math.PI){value=value -Math.PI;}
    while (value < -Math.PI){value=value +Math.PI;}
    return value;
}


function QuaternionToEulerAngle(quaternions_value)
{
  var euler={};
	// roll (x-axis rotation)
	var sinr_cosp = +2.0 * (quaternions_value.w * quaternions_value.x + quaternions_value.y * quaternions_value.z);
	var cosr_cosp = +1.0 - 2.0 * (quaternions_value.x * quaternions_value.x + quaternions_value.y * quaternions_value.y);
	euler.roll = normalizeRad(Math.atan2(sinr_cosp, cosr_cosp));

	// pitch (y-axis rotation)
	var sinp = +2.0 * (quaternions_value.w * quaternions_value.y - quaternions_value.z * quaternions_value.x);
	if ((sinp) >= 1){
    if(sinp>0){
      euler.pitch = normalizeRad(Math.PI/2);
    }else{
      euler.pitch = normalizeRad(-Math.PI/2);
    }
	}else{
		euler.pitch = normalizeRad(Math.sin(sinp));
  }
	// yaw (z-axis rotation)
	var siny_cosp = +2.0 * (quaternions_value.w * quaternions_value.z + quaternions_value.x * quaternions_value.y);
	var cosy_cosp = +1.0 - 2.0 * (quaternions_value.y * quaternions_value.y + quaternions_value.z * quaternions_value.z);
	euler.yaw = normalizeRad(Math.atan2(siny_cosp, cosy_cosp));
  return euler;
}

function mod(n, m) {
  return ((n % m) + m) % m;
}


</script>

