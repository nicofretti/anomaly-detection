createViewer('occupancygrid');
addGrid();
setTfClient(global_frame, '/robot/tf2_web_republisher', '/republish_tfs');
addRobot('robot/robot_description');
setMap(global_frame);
//interactiveMarkerClient('poi_interactive_marker');

var joystick;
$(document).ready(function () {
    joyStickZero = true;
    joystick = new VirtualJoystick({
        container: document.getElementById('imgJoystick'),
        mouseSupport: true,
    });

    setInterval(function () {
        if (joystick.deltaY() != 0 && joystick.deltaX() != 0 || !joyStickZero) {
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
            publish('/cmd_vel_mux/input/teleop', 'geometry_msgs/Twist', twist);
            if (joystick.deltaY() == 0 && joystick.deltaX() == 0) {
                joyStickZero = true;
            } else {
                joyStickZero = false;
            }
        }
        ;

    }, 1 / 30 * 1000);


    $('#tableCPU').DataTable({
        "bPaginate": false,
        "bFilter": false,
        "ordering": false,
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
        "columns": [
            {"data": "name"},
            {"data": "temp"}
        ]
    });

    $('#tableMemory').DataTable({
        "bPaginate": false,
        "bFilter": false,
        "ordering": false,
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
        "columns": [
            {"data": "name"},
            {"data": "state"},
            {"data": "input"},
            {"data": "output"},
            {"data": "received"},
            {"data": "transmited"},
            {"data": "rxerror"},
            {"data": "txerror"}
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

//
// MAP NAV STUFF
var mapping = false
var navigation = false
var localization = false
var map_server = false


function autorun_function() {
// assign function to onclick property of checkbox
    var aut = document.getElementById("autorun_default");
    aut.onclick = function () {
// access properties using this keyword
        if (this.checked) {
// if checked ...
//alert( this.value );
            enableLocalizationAutorun();
        } else {
// if not checked ...
            disableLocalizationAutorun();
        }
    };

}

/**
 * Function to start mapping
 * 
 */

function startMapping() {

    if (map_server == true) {
        window.alert("Error. map_server has to be switched off before running a mapping");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error. map_server has to be switched off before running a mapping";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        return;
    }
    if (mapping == true) {
        window.alert("Error: mapping already running");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: mapping already running";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    } else {
        callService('/map_nav_manager/start_mapping', 'std_srv/Trigger', {}, function (res) {
            console.log("startMapping - " + res.success + " - " + res.message);
            if (res.success) {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Mapping process response succesfull :" + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            } else {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Mapping process response not successfull: " + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            }
        });
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Send message to start mapping process.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;

//window.alert("Mapping process initiated. Press OK.");	            
    }
}

/**
 * Function to stop mapping
 * 
 */
function stopMapping() {

    if (mapping == false) {
        window.alert("Error: mapping process not initiated");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: mapping process not initiated";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    } else {
        callService('/map_nav_manager/stop_mapping', 'std_srv/Trigger', {}, function (res) {
            console.log("stopMapping - " + res.success + " - " + res.message);
            if (res.success) {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Mapping process response succesfull :" + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            } else {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Mapping process response not successfull: " + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            }
        });
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Send message to stop mapping process.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }
}


/**
 * Function to save map
 * 
 */
function saveMap() {
    if (mapping == false) {
        window.alert("Error. SLAM-Gmapping node not initiated");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error. SLAM-Gmapping node is not initiated.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    } else {
        var file_name = $('#filename').val();
        var checkbox_default = $('#checkbox_map_default').is(":checked");
        if (file_name == '') {
            window.alert("Error. The name of the map can not be empty.");
            document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error. The name of the map can not be empty.";
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        } else {

            callService('/map_nav_manager/save_map', 'map_nav_manager/SetFilename', {name: file_name, use_it_by_default: checkbox_default}, function (res) {
                console.log("saveMap - " + res.success + " - " + res.message);
//if(res.success){
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Save map process response succesfull.";
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
                /*}else{
                 document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info: Save map process response not successfull: " + res.message;
                 document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
                 }*/
            });
            document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Send save map whit the name " + file_name + ".";
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        }
    }
}

/**
 * Function to start Map Server
 */
function startMapServer() {
    if (mapping == true) {
        window.alert("Error. Mapping has to be off before running a map server");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: Mapping has to be off before running a map server";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        return;
    }

    if (map_server == true) {

        window.alert("Map server already running");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error: Map server already running";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    } else {

        var file_name = $('#filename').val();
        var checkbox_default = $('#checkbox_map_default').is(":checked");

        if (file_name == '') {
//window.alert("startMapServer: loading default map");
            document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: loading default map.";
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        }

        console.log(file_name);

        callService('/map_nav_manager/start_map_server', 'map_nav_manager/SetFilename', {name: file_name, use_it_by_default: checkbox_default}, function (res) {
            console.log("startMapServer - " + res.success + " - " + res.message);
//if(res.success){
            document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Map Server process response succesfull.";
            document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            /*}else{
             document.getElementById("monitor").value=document.getElementById("monitor").value+"\n Info: Map Server process response not successfull: " + res.message;
             document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
             }*/

        });
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Send message to star Map Server process with the map " + file_name + ".";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;

    }

}

/**
 * Function to stop Map Server
 */
function stopMapServer() {
    if (map_server == true) {
        callService('/map_nav_manager/stop_map_server', 'std_srv/Trigger', {}, function (res) {
            console.log("stopMapServer - " + res.success + " - " + res.message);
            if (res.success) {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Map Server process response succesfull :" + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            } else {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Map Server process response not successfull: " + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            }

        });
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Send message to stop Map Server process.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    } else {
        window.alert("Map Server not running");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Map Server stopped.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }
}

/**
 * Function to StarLocalization
 */
function startLocalization() {
    if (mapping == true) {
        window.alert("Error. Mapping has to be off before running localization node");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error. Mapping has to be off before running localization node.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        return;
    }

    if (map_server == false) {
        window.alert("Error. Localization node needs a map server node running");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error. Localization node needs a map server.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
        return;
    }

    if (localization == true) {

//window.alert("Localization already running");

    } else {

        callService('/map_nav_manager/start_localization', 'std_srv/Trigger', {}, function (res) {
            console.log("startLocalization - " + res.success + " - " + res.message);
            if (res.success) {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Localization process response succesfull :" + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            } else {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Localization response not successfull: " + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            }
        });
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Send message to start localization process.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }

}

/**
 * Function to stop Localization
 */

function stopLocalization() {

    if (localization == true) {

        callService('/map_nav_manager/stop_localization', 'std_srv/Trigger', {}, function (res) {
            console.log("stopLocalization - " + res.success + " - " + res.message);
            if (res.success) {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Localization process response succesfull :" + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            } else {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Localization response not successfull: " + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            }
        });
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Send message to stop localization process.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;

    } else {
//window.alert("Localization node is not running");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info. Localization node is not running.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }
}

/**
 * Function to disable Localization Autorun
 */
function disableLocalizationAutorun() {
    callService('/map_nav_manager/autorun_loc_and_nav', 'std_srv/SetBool', {data: false}, function (res) {
        console.log("disableLocalizationAutorun: Response received");
//window.alert("autorun disabled!");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info. Autorun disabled.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    });
}

/**
 * Function to star Navigation
 */
function startNavigation() {

    if (navigation == true) {
        window.alert("Error: navigation already running");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error. Navigation already running.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    } else {
        callService('/map_nav_manager/start_navigation', 'std_srv/Trigger', {}, function (res) {
            console.log("starNavigation - " + res.success + " - " + res.message);
            if (res.success) {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Navigation process response succesfull :" + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            } else {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Navigation response not successfull: " + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            }
        });
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Send message to start Navigation process.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;

    }

}

/**
 * Function to stop Navigation
 */
function stopNavigation() {

    if (navigation == false) {
        window.alert("Error: navigation not running");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Error. Navigation process is NOT running.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    } else {
        callService('/map_nav_manager/stop_navigation', 'std_srv/Trigger', {}, function (res) {
            console.log("stopNavigation - " + res.success + " - " + res.message);
            if (res.success) {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Navigation process response succesfull :" + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            } else {
                document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Navigation response not successfull: " + res.message;
                document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
            }
        });
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info: Send message to stop Navigation process.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    }
}

/**
 * Function to enable Localization Autorun
 */
function enableLocalizationAutorun() {
    callService('/map_nav_manager/autorun_loc_and_nav', 'std_srv/SetBool', {data: true}, function (res) {
        console.log("enableLocalizationAutorun: Response received");
//window.alert("autorun enabled!");
        document.getElementById("monitor").value = document.getElementById("monitor").value + "\n Info. Autorun enabled.";
        document.getElementById("monitor").scrollTop = document.getElementById("monitor").scrollHeight;
    });
}

/**
 * State map_nav_manager topic
 */
subscribe('/map_nav_manager/state', 'map_nav_manager/State', function (message) {
    mapping = message.mapping;
    navigation = message.navigation;
    localization = message.localization;
    map_server = message.map_server;

    if (mapping) {
//document.getElementById("mapping_status").style.color = "Green";       
        try {
            document.getElementById("button_save_map").disabled = false;
            document.getElementById("button_start_mapping").disabled = true;
            document.getElementById("button_stop_mapping").disabled = false;
        } catch (e) {

        }
    } else {
//document.getElementById("mapping_status").style.color = "Red";
        try {
            document.getElementById("button_save_map").disabled = true;
            document.getElementById("button_start_mapping").disabled = false;
            document.getElementById("button_stop_mapping").disabled = true;
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

});


// ODOMETRY
var odom_x = 0.0;
var odom_y = 0.0;
var odom_z = 0.0;
var odom_euler_x = 0.0;
var odom_euler_y = 0.0;
var odom_euler_z = 0.0;

// Odometry topic
subscribe('<? php echo $robot_dictionary["topic_odometry"]?>', 'nav_msgs/Odometry', function (message) {
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

});

function resetOdometry() {

}

// System_monitor
subscribe('/system_monitor/diagnostics', 'system_monitor/Diagnostic', function (message) {
    if (!readyOk) {
        return;
    }
//CPU
    var table_CPU = $('#tableCPU').DataTable();
    var cpu = message.diagCpuUsage.status.cores;
    table_CPU.clear();
    $.each(cpu, function (index, val_each) {
        var dataCPU = {};
        dataCPU["name"] = "CPU-" + val_each["id"].toString();
        dataCPU["speed"] = val_each["speed"].toString();
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
        dataTemp["name"] = "Core-" + val_each["id"].toString();
        dataTemp["temp"] = val_each["temp"].toString();
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
        dataMemory["total"] = val_each["total"].toString();
        dataMemory["used"] = val_each["used"].toString();
        dataMemory["free"] = val_each["free"].toString();
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
        dataHDD["size"] = val_each["size"].toString();
        dataHDD["use"] = val_each["use"].toString();
        dataHDD["available"] = val_each["available"].toString();
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
        dataNET["input"] = val_each["input"].toString();
        dataNET["output"] = val_each["output"].toString();
        dataNET["received"] = val_each["received"].toString();
        dataNET["transmited"] = val_each["transmitted"].toString();
        dataNET["rxerror"] = val_each["rxError"].toString();
        dataNET["txerror"] = val_each["txError"].toString();
        table_NET.row.add(dataNET);
    });
    table_NET.draw();
});


subscribe('/mobile_base/sensors/core', 'kobuki_msgs/SensorState', function (value) {
//Battery
    if (value.charger == "0") {
        $('#labelCharger').html("False");
    } else {
        $('#labelCharger').html("True");
    }
    $('#labelVoltage').html((parseInt(value.battery) / 10));
    if (value.over_current == "0") {
        $('#labelOverCurrent').html("False");
    } else {
        $('#labelOverCurrent').html("True");
    }
//Bumper
    var textBumper = "";
    if (value.bumper == "0") {
        textBumper = "None";
    } else {
        if (value.bumper & 1) {
            textBumper = " right";
        }
        if (value.bumper & 2) {
            textBumper += " center";

        }
        if (value.bumper & 4) {
            textBumper += " left";

        }
    }
    $('#labelBumper').html(textBumper);
//Wheels
    var textWheels = "";
    if (value.wheel_drop == "0") {
        textWheels = "None";
    } else {
        if (value.wheel_drop & 1) {
            textWheels = " right";
        }
        if (value.wheel_drop & 2) {
            textWheels += " left";

        }
    }
    $('#labelWhellDrop').html();
    $('#labelWhellDrop').html(textWheels);
    $('#labelLeftEnc').html(value.left_encoder);
    $('#labelRightEnc').html(value.right_encoder);
    $('#labelLeftPwm').html(value.left_pwm);
    $('#labelRightPwm').html(value.right_pwm);

//Ciff
    $('#labelCliff').html(value.cliff);
    $('#labelBottomRight').html(value.bottom[0]);
    $('#labelBottomCenter').html(value.bottom[1]);
    $('#labelBottomLeft').html(value.bottom[2]);
//Inpust
    $('#labelDigitalInp').html(value.digital_input);
    var textAnalogInput = "";
    $.each(value.analog_input, function (index, val_each) {
        textAnalogInput += val_each;
        textAnalogInput += " ";
    });
    $('#labelAnalogInp').html(textAnalogInput);

});

// called periodically to update the web elemets
function mainLoop() {
//console.log("mainloop");
// update odometry
    document.querySelector('#odometry_position_x span').innerHTML = odom_x;
    document.querySelector('#odometry_position_y span').innerHTML = odom_y;
    document.querySelector('#odometry_position_z span').innerHTML = odom_z;

    document.querySelector('#odometry_orientation_x span').innerHTML = odom_euler_x;
    document.querySelector('#odometry_orientation_y span').innerHTML = odom_euler_y;
    document.querySelector('#odometry_orientation_z span').innerHTML = odom_euler_z;
}

$(document).ready(function () {
// Sets trigger for map_nav check 
    autorun_function();
//call to mainLoop each x ms
    setInterval(mainLoop, main_loop_update_time);
});

