<div class="backgroundPageWeb" style="margin-top:60px">
    <div class="container-fluid">
        <div class="row">
            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                <div class="panel-group" id ="executingDiv">
                    <div class="panel panel-success">
                        <div class="panel-heading">
                            <h4 class="panel-title">
                                <a data-toggle="collapse" href="#executinCollapse">Running...</a>
                            </h4>
                        </div>
                        <div id="executinCollapse" class="panel-collapse collapse">
                            <div class="panel-body">
                                <div class="row">

                                    <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="complexExecutingStateDiv">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Running mission:</label><label id="complexMissionNameLabel"></label>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Current sequence in execution:</label><label id="complexMissionCurrentCommandLabel"></label>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="complexMissionNextCommandDiv">
                                                <label>Next sequences:</label><label id="complexMissionNextCommandLabel"></label>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="sequenceStateDiv">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Running sequence:</label><label id="sequenceNameLabel"></label>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Current action in execution:</label><label id="sequenceCurrentCommandLabel"></label>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="sequenceNextCommandDiv">
                                                <label>Next action:</label><label id="sequenceNextCommandLabel"></label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                <div class="panel-group" id ="errorPanelDiv">
                    <div class="panel panel-warning">
                        <div class="panel-heading" style="display:none">
                            <h4 class="panel-title">
                                <a data-toggle="collapse" href="#executinErrorCollapse">Last error in the execution</a>
                            </h4>
                        </div>
                        <div id="executinErrorCollapse" class="panel-collapse collapse">
                            <div class="panel-body">
                                <div class="row">
                                    <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="complexExecutingStateErrorDiv">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Last error in the mission:</label><label id="complexMissionErrorLabel"></label>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="sequenceStateErrorDiv">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Last error in the sequence:</label><label id="sequenceErrorLabel"></label>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                <div class="row">
                    <div class="col-xs-0 col-ms-0 col-md-4 col-xl-4">
                    </div>
                    <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                        <select class="form-control" id="sequenceMissionSelect">
                            <option value="-1">--Choose mission--</option>
                        </select>
                    </div>
                    <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                        <button type="button" class="btn btn-default btn-sm" onclick="stateLogicMaching(2)" id="createCommandMissionButton">
                            <span class="glyphicon glyphicon-plus"></span> Create mission
                        </button>
                        <button type="button" class="btn btn-default btn-sm" onclick="stateLogicMaching(3)" id="modifyCommandMissionButton">
                            <span class="glyphicon glyphicon-pencil"></span> Modify
                        </button>
                        <button type="button" class="btn btn-default btn-sm" id="deleteCommandMissionButton" onclick="deleteMission()">
                            <span class="glyphicon glyphicon-remove-circle"></span> Delete
                        </button>
                    </div>
                </div>
            </div>
        </div>
        <hr>
        <div class="row">
            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12"  id="commandMissionCompleteDiv">
                <div class="row" id="saveRunCancelButtonsDiv">
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-center">
                        <button class="btn btn-success" id="saveMissionButton" onclick="saveMission()">Save</button>
                    </div>
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-center">
                        <button class="btn btn-primary" id="runMissionButton" onclick="executeMission()">Execute mission</button>
                        <button class="btn btn-danger" id="cancelMissionButton" onclick="cancelMission()">Cancel mission</button>
                    </div>
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-center">
                        <button class="btn btn-danger" id="discardMissionButton" onclick="stateLogicMaching(0)">Cancel</button>
                    </div>
                </div>
                <hr>
                <div class="row" id="feedbackMissionDiv">

                </div>

                <div class="panel panel-danger" id="commandMissionDiv">
                    <div class="panel-heading">
                        <div class="form-group">
                            <label>Mission name: </label>
                            <label id="commandMissionLabel" type="text"></label>
                            <input type="text" class="form-control" id="commandMissionNameInput"></input>
                        </div>
                    </div>
                    <div class="panel-body">
                        <div clas="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <div class="row">
                                <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6">
                                    <div class="panel panel-danger" id="commandMissionStatesDiv">
                                        <div class="panel-heading">
                                            <div class="form-group">
                                                <label>States </label>
                                            </div>
                                        </div>
                                        <div class="panel-body">
                                            <div class="btn-group-vertical">
                                                <button type="button" class="btn btn-default" onclick="chargeNormalMission()">Normal</button>
                                                <button type="button" class="btn btn-default" onclick="chargeBatteryLowMission()">Low battery</button>
                                                <button type="button" class="btn btn-default" onclick="chargeErrorMission()">Error</button>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="panel panel-danger" id="sequenceInformationDiv">
                                        <div class="panel-heading">
                                            <div class="form-group">
                                                <div class="row">
                                                    <div class="col-xs-9 col-ms-9 col-md-9 col-xl-9">
                                                        <label>Sequence information</label>
                                                    </div>
                                                    <div class="col-xs-3 col-ms-3 col-md-3 col-xl-3 text-right">
                                                        <button id="upActionButton" class="btn btn-sm" style="color:black" onclick="moveUpAction()"><span class="glyphicon glyphicon-chevron-up"></span></button>
                                                        <button id="downActionButton" class="btn btn-sm" style="color:black" onclick="moveDownAction()"><span class="glyphicon glyphicon-chevron-down"></span></button>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="panel-body">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="SequenceDiv">
                                                <div class="row">
                                                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                        <button id="removeActionButton" class="btn btn-sm" onclick="deleteAction()"><span class="glyphicon glyphicon-trash"></span></button>
                                                        <label>Sequence <label id="sequenceLabel"></label> <label> when the system is on&nbsp</label><label id="stateOfSystemLabel"></label><label>&nbspstate.</label>
                                                    </div>

                                                    <div class="col-xs-12 col-ms-12 col-md-4 col-xl-4">
                                                        <button id="addCommandInSequenceButton" class="btn btn-success" onclick="addActionInSequence()">Add</button>
                                                    </div>
                                                    <div class="col-xs-12 col-ms-12 col-md-4 col-xl-4">
                                                        <button id="playCommandButton" class="btn btn-primary" onclick="executeSequence()">Execute</button>
                                                    </div>
                                                    <div class="col-xs-12 col-ms-12 col-md-4 col-xl-4">
                                                        <button id="discardCommandInSequenceButton" class="btn btn-danger" onclick="discardActionInSequence()">Discard</button>
                                                    </div>
                                                </div>

                                            </div>
                                        </div>
                                    </div>


                                    <div class="panel panel-danger" id="commandMissionAddSequenceDiv">
                                        <div class="panel-heading">
                                            <div class="form-group">
                                                <label>Add sequence</label>
                                            </div>
                                        </div>
                                        <div class="panel-body">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="addSequencesDiv">
                                                <div class="row" id="addSequenceListDiv">
                                                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                        <div class="row">
                                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                                <label>Add sequence:</label>
                                                                <input type="text" placeholder="Search.." id="searchSequenceInput" onkeyup="filterActionFunction()">
                                                            </div>
                                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                                <div class="btn-group-vertical" id="sequenceListDiv">
                                                                    <button type="button" class="btn btn-default">Test1</button>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                </div>
                                <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6">
                                    <div class="row">
                                        <div clas="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                            <label>Mission on&nbsp</label><label id="missionStateLabel">normal<label><label>&nbspstate:</label>
                                        </div>
                                        <div clas="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                            <div id="VisualizationMissionDiv" style="height: 500px">
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>

    let current_state_machine_ = 0;
    let mission_selected_ = -1;

    let mission_structure_ = {
        id: "",
        normal: "",
        battery_low: "",
        error: ""
    };

    let list_missions_ = [];

    let current_mission_selected_ = {};

    let gph_;
    let mission_information_visual_ = {};
    let current_visualization_type_ = "";

    const options = {
        manipulation: false,
        height: "90%",
        layout: {
            hierarchical: {
            enabled: true,
            levelSeparation: 150,
            },
        },
        physics: {
            hierarchicalRepulsion: {
            nodeDistance: 150,
            },
        },
    };


    function stateLogicMaching(new_state){
        if(new_state == 0){
            idleState()
        }
        else if (new_state == 1)
        {
            selectMissionState()
        }
        else if (new_state == 2)
        {
            createNewMissionState()
        }
        else if (new_state == 3)
        {
            modifyMissionState()
        }
        else{
            alertify.error("Error state")
            stateLogicMaching(0)
            return
        }
        current_state_machine_ = new_state
    }


    function idleState(){
        chargeListOfMissions();
        current_mission_selected_ = {};
        mission_information_visual_ = {};
        $("#createCommandMissionButton").show();
        $("#deleteCommandMissionButton").hide();
        $("#saveMissionButton").hide();
        $("#runMissionButton").hide();
        $("#discardMissionButton").hide();
        $("#saveRunCancelButtonsDiv").hide();
        $("#commandMissionLabel").text("");
        $("#commandMissionLabel").hide();
        $("#commandMissionNameInput").val("");
        $("#commandMissionNameInput").hide();
        $("#searchSequenceInput").val("");
        $("#addCommandInSequenceButton").hide();
        $("#playCommandButton").hide();
        $("#discardCommandInSequenceButton").hide();
        $("#sequenceInformationDiv").hide();
        $("#commandMissionAddSequenceDiv").hide();
        $("#commandMissionDiv").hide();
        $("#modifyCommandMissionButton").hide();
    }

    function selectMissionState(){
        $("#createCommandMissionButton").hide();
        $("#deleteCommandMissionButton").hide();
        $("#saveMissionButton").hide();
        $("#runMissionButton").show();
        $("#discardMissionButton").hide();
        $("#saveRunCancelButtonsDiv").show();
        $("#commandMissionLabel").show();
        $("#commandMissionNameInput").val("");
        $("#commandMissionNameInput").hide();
        $("#searchSequenceInput").val("");
        $("#addCommandInSequenceButton").hide();
        $("#playCommandButton").show();
        $("#discardCommandInSequenceButton").hide();
        $("#sequenceInformationDiv").hide();
        $("#commandMissionAddSequenceDiv").hide();
        $("#commandMissionDiv").show();
        $("#modifyCommandMissionButton").show();
    }

    function modifyMissionState(){
        chargeSequences();
        $("#createCommandMissionButton").hide();
        $("#deleteCommandMissionButton").show();
        $("#saveMissionButton").show();
        $("#runMissionButton").hide();
        $("#discardMissionButton").show();
        $("#saveRunCancelButtonsDiv").show();
        $("#commandMissionLabel").hide();
        $("#commandMissionNameInput").val($("#sequenceMissionSelect").val());
        $("#commandMissionNameInput").hide();
        $("#searchSequenceInput").val("");
        $("#addCommandInSequenceButton").show();
        $("#playCommandButton").show();
        $("#discardCommandInSequenceButton").show();
        $("#sequenceInformationDiv").hide();
        $("#commandMissionAddSequenceDiv").show();
        $("#commandMissionDiv").show();
        $("#modifyCommandMissionButton").hide();
    }

    function createNewMissionState(){
        chargeSequences();
        initVisualization();
        current_mission_selected_ = Object.assign({}, mission_structure_);
        $("#createCommandMissionButton").hide();
        $("#deleteCommandMissionButton").hide();
        $("#saveMissionButton").show();
        $("#runMissionButton").hide();
        $("#discardMissionButton").show();
        $("#saveRunCancelButtonsDiv").show();
        $("#commandMissionLabel").hide();
        $("#commandMissionNameInput").val("");
        $("#commandMissionNameInput").show();
        $("#searchSequenceInput").val("");
        $("#addCommandInSequenceButton").show();
        $("#playCommandButton").show();
        $("#discardCommandInSequenceButton").show();
        $("#sequenceInformationDiv").hide();
        $("#commandMissionAddSequenceDiv").show();
        $("#commandMissionDiv").show();
        $("#modifyCommandMissionButton").hide();

    }


    function chargeListOfMissions(){
        callService('<?php echo $robot_dictionary["service_complex_sequencer_get_sequences"]?>', 'robot_simple_command_manager_msgs/GetCommandManagerList', {}, function (res) {
            mission_selected_ = -1;
            list_missions_ = [];
            $('#sequenceMissionSelect').empty();
            $('#sequenceMissionSelect').append($('<option>', {
                value: -1,
                text: "--Choose mission--"
                }));
            $('[name=options]').val(-1);
            if(res.success != true){
                alertify.error("Error to charge command sequences. Is it the command sequences running?");
                return;
            }
            else{
                jQuery.each(res["list"]["commands"], function( indx, val ) {
                    id_split = val["id"].split("_");
                    if(!(id_split[id_split.length-1]=="BATLOW" || id_split[id_split.length-1]=="ERROR")){
                        var struct = Object.assign({}, mission_structure_);
                        $('#sequenceMissionSelect').append($('<option>', {
                            value: val["id"],
                            text: val["id"]
                        }));
                        struct["id"] = val["id"];
                        struct["normal"] = val["command"];
                        var i;
                        for(i=0; i < res["list"]["commands"].length; i++){
                            if(val["id"] + "_BATLOW" == res["list"]["commands"][i]["id"]){
                                struct["battery_low"] = res["list"]["commands"][i]["command"];
                            }else if(val["id"] + "_ERROR" == res["list"]["commands"][i]["id"]){
                                struct["error"] = res["list"]["commands"][i]["command"];
                            }
                        }
                        list_missions_.push(struct);
                    }
                });
            }
        });
    }

    $("#sequenceMissionSelect" ).change(function () {
        $( "#sequenceMissionSelect option:selected" ).each(function() {
            mission_selected_ = $( this ).val();
            if(mission_selected_!=-1){
                $("#commandMissionLabel").text($( this ).val());
            }
        });
        if(mission_selected_ == -1){
            stateLogicMaching(0);
        }else{
            current_mission_selected_ = Object.assign({}, mission_structure_);

            jQuery.each(list_missions_, function( indx, val ) {
                if(val["id"] == mission_selected_){
                    current_mission_selected_ = val;
                }
            });
            chargeNormalMission();
            stateLogicMaching(1);
        }
    }).change();

    function initVisualization(label = "Normal"){
        if(label.toLowerCase() == "normal"){
            current_visualization_type_ = "Normal";
        }else if(label.toLowerCase() == "battery low"){
            current_visualization_type_ = "battery_low";
        }else if(label.toLowerCase() == "error"){
            current_visualization_type_ = "error";
        }else{
            current_visualization_type_ = "unknown";
            label = "Unknown"
        }
        $("#sequenceInformationDiv").hide();
        $("#missionStateLabel").text(label.toLowerCase());
        if(gph_ != null){
            gph_.destroy();
        }
        container = document.getElementById("VisualizationMissionDiv");

        mission_information_visual_ = {
            nodes: [
                {
                    id: "Init_01_____0",
                    size: 100,
                    label: label,
                    color: "#0000FF",
                    shape: "cicle",
                    font: { face: "monospace", align: "left" }
                }
            ],
            edges: [

            ]
        }

        gph_ = new vis.Network(container, mission_information_visual_, options);
        gph_.on("click", function(params){

            index_sequence_select = parseInt(params["nodes"][0].split("_01_____")[1]);
            value_sequence_select = params["nodes"][0].split("_01_____")[0];
            if(index_sequence_select == 0){
                $("#sequenceInformationDiv").hide();
            }else{
                if(current_state_machine_ == 3 || current_state_machine_ == 2){
                    if(index_sequence_select!=1){
                        $("#upActionButton").show();
                    }else{
                        $("#upActionButton").hide();
                    }
                    var split = current_mission_selected_[current_visualization_type_.toLowerCase()].split(";");
                    if(index_sequence_select==split.length){
                        $("#downActionButton").hide();
                    }else{
                        $("#downActionButton").show();
                    }
                    $("#removeActionButton").show();
                }else{
                    $("#upActionButton").hide();
                    $("#downActionButton").hide();
                    $("#removeActionButton").hide();
                }
                $("#sequenceLabel").text(value_sequence_select);
                $("#stateOfSystemLabel").text($("#missionStateLabel").text());
                $("#addCommandInSequenceButton").hide();
                $("#playCommandButton").show();
                $("#discardCommandInSequenceButton").hide();
                $("#sequenceInformationDiv").show();
            }

        });
    };

    function moveUpAction()
    {
        current_mission_selected_[current_visualization_type_.toLowerCase()] = current_mission_selected_[current_visualization_type_.toLowerCase()].replaceAll(' ;', ';')
        current_mission_selected_[current_visualization_type_.toLowerCase()] = current_mission_selected_[current_visualization_type_.toLowerCase()].replaceAll('; ', ';')
        var split = current_mission_selected_[current_visualization_type_.toLowerCase()].split(";");
        if(index_sequence_select<=1 || split.length < 2)
        {
            alertify.error("Error to up action. Id: " + index_sequence_select + ". Type: " + current_visualization_type_);
            return;
        }
        var new_sequence = "";
        //for(i=split.length-1; i >= 0; i--)
        for (i=0; i < split.length; i++)
        {
            if(i==index_sequence_select-2){
                new_sequence = new_sequence + split[i+1] + ";" ;
                new_sequence = new_sequence + split[i] + ";" ;
                i++;
            }else
            {
                new_sequence = new_sequence + split[i] + ";" ;
            }
        }
        new_sequence = new_sequence.substring(0, new_sequence.length - 1);
        console.log(new_sequence);
        current_mission_selected_[current_visualization_type_.toLowerCase()] = new_sequence;
        if(current_visualization_type_.toLowerCase() == "normal"){
            chargeNormalMission();
        }else if(current_visualization_type_.toLowerCase() == "battery_low"){
            chargeBatteryLowMission();
        }else if(current_visualization_type_.toLowerCase() == "error"){
            chargeErrorMission()
        }
    }

    function moveDownAction()
    {
        current_mission_selected_[current_visualization_type_.toLowerCase()] = current_mission_selected_[current_visualization_type_.toLowerCase()].replaceAll(' ;', ';')
        current_mission_selected_[current_visualization_type_.toLowerCase()] = current_mission_selected_[current_visualization_type_.toLowerCase()].replaceAll('; ', ';')
        var split = current_mission_selected_[current_visualization_type_.toLowerCase()].split(";");
        if(index_sequence_select==split.length || split.length < 2)
        {
            alertify.error("Error to down action. Id: " + index_sequence_select + ". Type: " + current_visualization_type_);
            return;
        }
        var new_sequence = "";
        //for(i=split.length-1; i >= 0; i--)
        for (i=0; i < split.length; i++)
        {
            if(i==index_sequence_select-1){
                new_sequence = new_sequence + split[i+1] + ";" ;
                new_sequence = new_sequence + split[i] + ";" ;
                i++;
            }else
            {
                new_sequence = new_sequence + split[i] + ";" ;
            }
        }
        new_sequence = new_sequence.substring(0, new_sequence.length - 1);
        console.log(new_sequence);
        current_mission_selected_[current_visualization_type_.toLowerCase()] = new_sequence;
        if(current_visualization_type_.toLowerCase() == "normal"){
            chargeNormalMission();
        }else if(current_visualization_type_.toLowerCase() == "battery low"){
            chargeBatteryLowMission();
        }else if(current_visualization_type_.toLowerCase() == "error"){
            chargeErrorMission()
        }
    }

    function deleteAction()
    {
        if(index_sequence_select<0)
        {
            alertify.error("Error to delete action. Id: " + index_sequence_select + ". Type: " + current_visualization_type_);
            return;
        }
        alertify.confirm('Delete sequence','Do you want to remove the sequence from the mission list?', function(){
            console.log(current_mission_selected_[current_visualization_type_.toLowerCase()]);
            var new_sequence = "";
            current_mission_selected_[current_visualization_type_.toLowerCase()] = current_mission_selected_[current_visualization_type_.toLowerCase()].replaceAll(' ;', ';')
            current_mission_selected_[current_visualization_type_.toLowerCase()] = current_mission_selected_[current_visualization_type_.toLowerCase()].replaceAll('; ', ';')
            var split = current_mission_selected_[current_visualization_type_.toLowerCase()].split(";");
            for (i=0; i < split.length; i++)
            {
                if(i != index_sequence_select -1){
                    new_sequence = new_sequence + split[i] + ";" ;
                }
            }
            new_sequence = new_sequence.substring(0, new_sequence.length - 1);
            console.log(new_sequence);
            current_mission_selected_[current_visualization_type_.toLowerCase()] = new_sequence;
            if(current_visualization_type_.toLowerCase() == "normal"){
                chargeNormalMission();
            }else if(current_visualization_type_.toLowerCase() == "battery low"){
                chargeBatteryLowMission();
            }else if(current_visualization_type_.toLowerCase() == "error"){
                chargeErrorMission()
            }
        }
        , function(){
            alertify.success('Discarded')}
        );
    }

    function addElementInVisualization(command){
        if(command==""){
            return true;
        }
        last_index = (mission_information_visual_.nodes.length) -1;
        id_command = command+"_01_____"+(last_index+1).toString();;
        label_in_node = command;

        new_edge = {
            from: mission_information_visual_.nodes[last_index]["id"],
            to: id_command,
            arrows: "to",
            physics: false,
            smooth: { type: "cubicBezier" }
        };

        new_node = {
           id: id_command,
           size: 400,
           color: "#FFCFCF",
           shape: "box",
           font: { face: "monospace", align: "left" },
           label: label_in_node
        }


        mission_information_visual_["nodes"].push(new_node)
        mission_information_visual_["edges"].push(new_edge)
        gph_.setData(mission_information_visual_);
        return true;
    }

    function chargeNormalMission(){
        initVisualization("Normal");
        if(current_mission_selected_["normal"].split(";") == 0){
            return;
        }
        jQuery.each(current_mission_selected_["normal"].split(";"), function( normal_mission_indx, normal_mission_val ) {
            addElementInVisualization(normal_mission_val);
        });
    }

    function chargeBatteryLowMission(){
        initVisualization("Battery low");
        if(current_mission_selected_["battery_low"].split(";") == 0){
            return;
        }
        jQuery.each(current_mission_selected_["battery_low"].split(";"), function( battery_low_mission_indx, battery_low_mission_val ) {
            addElementInVisualization(battery_low_mission_val);
        });
    }

    function chargeErrorMission(){
        initVisualization("Error");
        if(current_mission_selected_["error"].split(";") == 0){
            return;
        }
        jQuery.each(current_mission_selected_["error"].split(";"), function( error_mission_indx, error_mission_val ) {
            addElementInVisualization(error_mission_val);
        });
    }

    function runMission(){
        alertify.confirm('Run mission', 'Do you want run the mission?', function(){executeMission() }, function(){alertify.success('Discarded')});
    }

    function executeMission(){
        if(mission_selected_==-1){
            alertify.error("Error to charge mission.");
            return;
        }
        var complex_sequencer_srv={
            command: mission_selected_
        }
        console.log(complex_sequencer_srv);
        callService('<?php echo $robot_dictionary["service_complex_sequencer_command"]?>', 'robot_simple_command_manager_msgs/SetCommandString', complex_sequencer_srv, function (res) {
            if(res["ret"]["success"]){
                alertify.success('Exectuting mission');
            }else{
                alertify.error("Error to start mission. Error: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
                return;
            }
        });
    }

    function cancelMission(){
        srv = {};
        callService('<?php echo $robot_dictionary["service_complex_sequencer_cancel"]?>', 'std_srvs/Trigger', srv, function (res) {
            if(res["success"]){
                alertify.success('Cancelled mission');
            }else{
                alertify.error("Error to cancel mission. Error: " + res["message"]);
                return;
            }
        });
    }

    function chargeSequences(){
        var list_sequences_element = document.getElementById("sequenceListDiv");
        list_sequences_element.innerHTML = "";
        callService('<?php echo $robot_dictionary["service_command_sequencer_get_sequences"]?>', 'robot_simple_command_manager_msgs/GetCommandManagerList', {}, function (res) {
            jQuery.each(res["list"]["commands"], function( sequence_index, sequence_foreach ){
                var button_element = document.createElement('button');
                button_element.id="sequence_" + sequence_foreach["id"] + "Button";
                button_element.onclick=function(){
                    addSequence(sequence_foreach["id"]);
                    $("#upActionButton").hide();
                    $("#downActionButton").hide();
                    $("#removeActionButton").hide();
                    $("#sequenceInformationDiv").show();
                }
                button_element.className = "btn btn-default btn-block";
                button_element.textContent =  sequence_foreach["id"];
                list_sequences_element.appendChild(button_element);
            });
        });
    }

    function filterActionFunction() {
        var input, filter, ul, li, a, i;
        input = document.getElementById("searchSequenceInput");
        filter = input.value.toUpperCase();
        div = document.getElementById("sequenceListDiv");
        a = div.getElementsByTagName("button");
        for (i = 0; i < a.length; i++) {
            txtValue = a[i].textContent || a[i].innerText;
            if (txtValue.toUpperCase().indexOf(filter) > -1) {
            a[i].style.display = "";
            } else {
            a[i].style.display = "none";
            }
        }
    }

    function addSequence(sequence_id){
        $("#stateOfSystemLabel").text(current_visualization_type_.toLowerCase());
        $("#sequenceLabel").text(sequence_id);
        $("#addCommandInSequenceButton").show();
        $("#playCommandButton").show();
        $("#discardCommandInSequenceButton").show();
    }

    function executeSequence(){
        sequence_id = $("#sequenceLabel").text();
        if(sequence_id == ""){
            alertify.error('Error to add sequence');
            return;
        }
        alertify.confirm('Execute ' + sequence_id, 'Do you want execute the sequence ' + sequence_id + "?", function(){
            command = sequence_id;
            srv_command = {
                command: command
            }
            callService('<?php echo $robot_dictionary["service_command_sequencer_command"]?>', 'robot_simple_command_manager_msgs/SetCommandString', srv_command, function (res) {
                if(res["ret"]["success"]){
                    alertify.success('Exectuting sequence');
                }else{
                    alertify.error(res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
                }

            });
        }
        , function(){
            alertify.success('Discarded')}
        );
    }

    function addActionInSequence()
    {
        sequence_id = $("#sequenceLabel").text();
        if(current_visualization_type_ == "" || sequence_id == ""){
            alertify.error('Error to add sequence');
            return;
        }
        alertify.confirm('Add ' + sequence_id, 'Do you want add ' + sequence_id + " on " + current_visualization_type_.toLowerCase() + "?", function(){
            if(current_mission_selected_[current_visualization_type_.toLowerCase()]!=""){
                current_mission_selected_[current_visualization_type_.toLowerCase()] += "; ";
            }
            current_mission_selected_[current_visualization_type_.toLowerCase()] += sequence_id;
            addElementInVisualization(sequence_id);
            $("#sequenceInformationDiv").hide();
        }
        , function(){
            alertify.success('Discarded')}
        );
    }

    function discardActionInSequence(){
        $("#sequenceInformationDiv").hide();
    }

    function executeMission(){
        mission = mission_selected_;
        if(mission == "" || mission== -1){
            alertify.error('Error to execute mission');
            return;
        }
        alertify.confirm('Execute ' + mission, 'Do you want execute the mission ' + mission + "?", function(){
            command = mission;
            srv_command = {
                command: command
            }
            callService('<?php echo $robot_dictionary["service_complex_sequencer_command"]?>', 'robot_simple_command_manager_msgs/SetCommandString', srv_command, function (res) {
                if(res["ret"]["success"]){
                    alertify.success('Exectuting mission');
                }else{
                    alertify.error(res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
                }

            });
        }
        , function(){
            alertify.success('Discarded')}
        );
    }

    function saveMission(){
        if($("#commandMissionNameInput").val()==""){
            alertify.error("It is necessary to give a name to the mission");
            return;
        }
        var normal_command = current_mission_selected_["normal"];
        var battery_low_command = current_mission_selected_["battery_low"];
        var error_command = current_mission_selected_["error"];
        if(normal_command == "" || normal_command == null){
            alertify.error("It is necessary to add at least a normal state description");
            return;
        }
        var create_normal_mission_srv = {
            Command: {
                id: $("#commandMissionNameInput").val(),
                command: normal_command
            }
        };
        var create_batlow_mission_srv = {
            Command: {
                id: $("#commandMissionNameInput").val()+ "_BATLOW",
                command: battery_low_command
            }
        };
        var create_error_mission_srv = {
            Command: {
                id: $("#commandMissionNameInput").val() + "_ERROR",
                command: error_command
            }
        };
        callService('<?php echo $robot_dictionary["service_complex_sequencer_add_sequence"]?>', 'robot_simple_command_manager_msgs/ManageCommandManager', create_normal_mission_srv, function (res) {
            if(res["ret"]["success"]){
                alertify.success('Saved: ' + $("#commandMissionNameInput").val());
                stateLogicMaching(0);
            }else{
                alertify.error("Error, to save mission. Message: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
            }
        });
        callService('<?php echo $robot_dictionary["service_complex_sequencer_add_sequence"]?>', 'robot_simple_command_manager_msgs/ManageCommandManager', create_batlow_mission_srv, function (res) {
            if(!res["ret"]["success"]){
                alertify.error("Error, to save mission when it is in battery low state. Message: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
            }else{
                stateLogicMaching(0);
            }
        });
        callService('<?php echo $robot_dictionary["service_complex_sequencer_add_sequence"]?>', 'robot_simple_command_manager_msgs/ManageCommandManager', create_error_mission_srv, function (res) {
            if(!res["ret"]["success"]){
                alertify.error("Error, to save mission when it is in error state. Message: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
            }else{
                stateLogicMaching(0);
            }
        });

    }

    function deleteMission(){
        alertify.confirm('Remove ' + $("#commandMissionLabel").text(), 'Do you want remove the mission?', function(){removeMission() }, function(){alertify.success('Discarded')});
    }

    function removeMission(){

        if($("#commandMissionLabel").text()==""){
            alertify.error("It is necessary to give a name to the mission");
            return;
        }

        var delete_normal_mission_srv = {
            Command: {
                id: $("#commandMissionLabel").text(),
                command: ""
            }
        };var delete_error_mission_srv = {
            Command: {
                id: $("#commandMissionLabel").text() + "_BATLOW",
                command: ""
            }
        };
        var delete_error_mission_srv = {
            Command: {
                id: $("#commandMissionLabel").text() + "_ERROR",
                command: ""
            }
        };

        callService('<?php echo $robot_dictionary["service_complex_sequencer_remove_sequence"]?>', 'robot_simple_command_manager_msgs/ManageCommandManager', delete_normal_mission_srv, function (res) {
            if(res["ret"]["success"]){
                alertify.success('Deleted: ' + $("#commandMissionLabel").text());
                stateLogicMaching(0);
            }else{
                alertify.error("Error, to remove mission. Message: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
            }
        });
        callService('<?php echo $robot_dictionary["service_complex_sequencer_remove_sequence"]?>', 'robot_simple_command_manager_msgs/ManageCommandManager', delete_batlow_mission_srv, function (res) {
            if(!res["ret"]["success"]){
                alertify.error("Error, to save mission when it is in battery low state. Message: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
            }else{
                stateLogicMaching(0);
            }
        });
        callService('<?php echo $robot_dictionary["service_complex_sequencer_remove_sequence"]?>', 'robot_simple_command_manager_msgs/ManageCommandManager', delete_error_mission_srv, function (res) {
            if(!res["ret"]["success"]){
                alertify.error("Error, to save mission when it is in error state. Message: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
            }else{
                stateLogicMaching(0);
            }
        });
    }

    stateLogicMaching(0);



    subscribe('<?php echo $robot_dictionary["topic_complex_sequencer_status"]?>', 'robot_simple_command_manager_msgs/SequencerStatus', function (message) {
        if(message.current_sequence.commands.length!=0){
            $("#complexMissionNameLabel").text(message.current_sequence.name);
            $("#complexMissionCurrentCommandLabel").text(message.current_sequence.current_command);
            if(message.current_sequence.command_index == message.current_sequence.commands.length || message.current_sequence.commands.length==0){
                $("#complexMissionNextCommandLabel").text("");
                $("#complexMissionNextCommandDiv").hide();
            }else{
                list_sequences = ""
                for(i = message.current_sequence.command_index; i < message.current_sequence.commands.length; i++){
                    list_sequences += message.current_sequence.commands[i] + " ";
                }
                $("#complexMissionNextCommandLabel").text(list_sequences);
                $("#complexMissionNextCommandDiv").show();
            }
            $("#complexExecutingStateDiv").show();
            $("#executingDiv").show();
        }else{
            $("#complexMissionNameLabel").text("");
            $("#complexExecutingStateDiv").hide();
            if($("#sequenceNameLabel").text()==""){
                $("#executingDiv").hide();
            }
        }
        $("#complexMissionErrorLabel").text(message.last_error);
        if(message.last_error==""){
            $("#complexExecutingStateErrorDiv").hide();
            if ($("#sequenceErrorLabel").text()==""){
                $("#errorPanelDiv").hide()
            }
        }else{
            $("#complexExecutingStateErrorDiv").show();
            $("#errorPanelDiv").show()
        }
    });

    subscribe('<?php echo $robot_dictionary["topic_command_sequencer_status"]?>', 'robot_simple_command_manager_msgs/SequencerStatus', function (message) {
        if(message.current_sequence.commands.length!=0){
            $("#sequenceNameLabel").text(message.current_sequence.name);
            $("#sequenceCurrentCommandLabel").text(message.current_sequence.current_command);
            if(message.current_sequence.command_index == message.current_sequence.commands.length || message.current_sequence.commands.length==0){
                $("#sequenceNextCommandLabel").text("");
                $("#sequenceNextCommandDiv").hide();
            }else{
                list_sequences = ""
                for(i = message.current_sequence.command_index; i < message.current_sequence.commands.length; i++){
                    list_sequences += message.current_sequence.commands[i] + " ";
                }
                $("#sequenceNextCommandLabel").text(list_sequences);
                $("#sequenceNextCommandDiv").show();
            }
            $("#sequenceStateDiv").show();
            $("#executingDiv").show();
        }else{
            $("#sequenceNameLabel").text("");
            $("#sequenceStateDiv").hide();
            if($("#complexMissionNameLabel").text()==""){
                $("#executingDiv").hide();
            }
        }
        $("#sequenceErrorLabel").text(message.last_error);
        if(message.last_error==""){
            $("#sequenceStateErrorDiv").hide();
            if ($("#complexExecutingStateErrorDiv").text()==""){
                $("#errorPanelDiv").hide()
            }
        }else{
            $("#sequenceStateErrorDiv").show();
            $("#errorPanelDiv").show()
        }
    });


</script>
