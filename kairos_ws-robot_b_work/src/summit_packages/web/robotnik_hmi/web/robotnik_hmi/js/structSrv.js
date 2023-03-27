/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

function getStructOdometry (){
    return {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        orientation: 0.0
    }
}

function getInsertMission(){
    return {
        mission:
                {
                    id: 0,
                    robot_id: -1,
                    charge: false,
                    load: false,
                    unload: false,
                    canceled: false,
                    error_msg: '',
                    node_or: [0],
                    node_dest: [0],
                    node_or_action: '',
                    node_dest_action: '',
                    time_start: '',
                    time_insert: '',
                    time_end: '',

                }
    }
}


function getCancelMission (){
    return {
        mission:0,
        robot: 0,
        msg: ''
    }
}

function getRobotNodeTasks (){
    return {
        robot: 0
    }
}

function getRobotBlockedNodes (){
    return {
        robot_id: 0
    }
}
