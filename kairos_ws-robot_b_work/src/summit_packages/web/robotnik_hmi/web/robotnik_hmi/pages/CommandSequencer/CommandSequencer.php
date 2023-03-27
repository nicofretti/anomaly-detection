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


                                    <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6" id="sequenceStateDiv">
                                        <div class="row">
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Running sequence:&nbsp;</label><label id="sequenceNameLabel"></label>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                                <label>Current command:&nbsp;</label><label id="sequenceCurrentCommandLabel"></label>
                                            </div>
                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="sequenceNextCommandDiv">
                                                <label>Next action:&nbsp;</label><label id="sequenceNextCommandLabel"></label>
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
                        <div class="panel-heading">
                            <h4 class="panel-title">
                                <a data-toggle="collapse" href="#executinErrorCollapse"></a>
                            </h4>
                        </div>
                        <div id="executinErrorCollapse" class="panel-collapse collapse">
                            <div class="panel-body">
                                <div class="row">
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
                        <select class="form-control" id="sequenceCommandSelect">
                            <option value="-1">--Choose command--</option>
                        </select>
                    </div>
                    <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                        <button type="button" class="btn btn-default btn-sm" id="createCommandSequenceButton" onclick="stateLogicMaching(2)">
                            <span class="glyphicon glyphicon-plus"></span> Create sequence
                        </button>
                        <button type="button" class="btn btn-default btn-sm" id="modifyCommandSequenceButton" onclick="stateLogicMaching(3)">
                            <span class="glyphicon glyphicon-pencil"></span> Modify
                        </button>
                        <button type="button" class="btn btn-default btn-sm" id="deleteCommandSequenceButton"  onclick="deleteSequence()">
                            <span class="glyphicon glyphicon-remove-circle"></span> Delete
                        </button>
                    </div>
                </div>
            </div>
        </div>
        <hr>
        <div class="row">
            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12"  id="commandSequenceDiv">
                <div class="row" id="saveRunCancelButtonsDiv">
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-center">
                        <button class="btn btn-success" id="saveSequenceButton"  onclick="saveSequence()">Save</button>
                    </div>
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-center">
                        <button class="btn btn-primary" id="runSequenceButton" onclick="runSequence()">Run</button>
                        <button class="btn btn-danger" id="runSequenceButton" onclick="cancelSequence()">Cancel</button>
                    </div>
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-center">
                        <button class="btn btn-danger" id="discardSequenceButton" onclick="stateLogicMaching(0)">Discard</button>
                    </div>
                </div>
                <div class="panel panel-danger" id="commandSequenceDiv">
                    <div class="panel-heading">
                        <div class="form-group">
                            <label>Command sequence name: </label>
                            <label id="commandSequenceLabel" type="text"></label>
                            <input type="text" class="form-control" id="commandSequenceNameInput"></input>
                        </div>
                    </div>
                    <div class="panel-body">
                        <div clas="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                            <div class="row">
                                <div class="col-xs-6 col-ms-6 col-md-3 col-xl-3" style="background: #FFFFFF;">
                                    <div class="row" id="newActionDiv">
                                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="listCommandAcctionToAddDiv">
                                            <div id="listActionsWithSearchDiv">
                                                <label>Add actions:</label>
                                                <input type="text" placeholder="Search.." id="searchActionInput" onkeyup="filterFunction()">
                                                <div id="listActionsDiv">
                                                    <a>None</a>
                                                </div>
                                            </div>

                                        </div>


                                    </div>

                                    <div class="row" id="actionInfoDiv">


                                        <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12"  id="commandParamsDiv">
                                            <div class="panel-group">
                                            <label for="listGroup">Previous action:</label><label id="previousActionLabel"></label>
                                                <div class="panel panel-default">
                                                    <div class="panel-heading">
                                                        <div class="row">
                                                            <div class="col-xs-9 col-ms-9 col-md-9 col-xl-9">
                                                                <h4 class="panel-title">
                                                                    <button id="removeActionButton" class="btn btn-sm" onclick="deleteAction()"><span class="glyphicon glyphicon-trash"></span></button>
                                                                    <label>Params of </label> <label id="paramsOfLabel"></label></a>
                                                                </h4>
                                                            </div>
                                                            <div class="col-xs-3 col-ms-3 col-md-3 col-xl-3">
                                                                <button id="upActionButton" class="btn btn-sm" onclick="moveUpAction()"><span class="glyphicon glyphicon-chevron-up"></span></button>
                                                                <button id="downActionButton" class="btn btn-sm" onclick="moveDownAction()"><span class="glyphicon glyphicon-chevron-down"></span></button>
                                                            </div>
                                                        </div>

                                                    </div>
                                                    <div class="panel-body">
                                                        <div class="row">
                                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="panelCommands">

                                                            </div>
                                                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="commandButtonsDiv">
                                                                <div class="row">
                                                                    <div class="col-xs-12 col-ms-12 col-md-4 col-xl-4">
                                                                        <button id="addCommandInSequenceButton" class="btn btn-success" onclick="addActionInSequence()">Add</button>
                                                                    </div>
                                                                    <div class="col-xs-12 col-ms-12 col-md-4 col-xl-4">
                                                                        <button id="playCommandButton" class="btn btn-primary" onclick="playCommand()">Run</button>
                                                                    </div>
                                                                    <div class="col-xs-12 col-ms-12 col-md-4 col-xl-4">
                                                                        <button id="discardCommandInSequenceButton" class="btn btn-danger" onclick="discardActionInSequence()">Discard</button>
                                                                    </div>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </div>

                                                    <div id="params_elem" hidden>
                                                        <div id="StringDiv">
                                                            <div class="form-group">
                                                                <label for="textGroup">Text:</label>
                                                                <input type="text" class="form-control" id="textGroup">
                                                            </div>
                                                        </div>

                                                        <div id="IntegerNumberDiv">
                                                            <div class="form-group">
                                                                <label for="numberGroup">Integer number:</label>
                                                                <input type="number" step=1 class="form-control" id="numberGroup" onkeypress="return isNumber(event)" onpaste="return false;">
                                                            </div>
                                                        </div>

                                                        <div id="DoubleNumberDiv">
                                                            <div class="form-group">
                                                                <label for="numberGroup">Double number:</label>
                                                                <input type="number" step="any" class="form-control" id="numberGroup" onkeypress="return isFloat(event)" onpaste="return false;">
                                                            </div>
                                                        </div>
                                                        <div id="BooleanDiv">
                                                            <div class="form-check">
                                                                <label class="form-check-label" for="booleanInput">
                                                                    Default checkbox
                                                                </label>
                                                                <input class="form-check-input" type="checkbox" value="" id="booleanInput">

                                                            </div>
                                                        </div>
                                                        <!--div id="BooleanDiv">
                                                            <label id="booleanLabel">Option</label><input id="booleanInput" type="checkbox" value="">
                                                        </div-->
                                                        <div id="ListDiv">
                                                            <div class="form-group">
                                                                <label for="listGroup">Select:</label>
                                                                <select class="form-control" id="listGroup">
                                                                    <!--option>The service does not respond</option>
                                                                    <option>2</option>
                                                                    <option>3</option>
                                                                    <option>4</option-->
                                                                </select>
                                                            </div>
                                                        </div>
                                                        <!--div class="row" id="addDiscardDeleteCommand">
                                                            <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6 text-center" id="addCommandInSequenceButton">
                                                                <button class="btn btn-success">Add</button>
                                                            </div>
                                                            <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6 text-center" id="discardCommandButton">
                                                                <button class="btn btn-success">Discard</button>
                                                            </div>
                                                            <div class="col-xs-12 col-ms-12 col-md-6 col-xl-6 text-center" id="deleteCommandButton">
                                                                <button class="btn btn-success" onclick="deleteSequence()">Delete</button>
                                                            </div>
                                                        </div-->
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                </div>

                                <div class="col-xs-6 col-ms-6 col-md-9 col-xl-9">
                                    <div class="row">
                                        <label>Sequence</label>
                                        <div id="VisualizationActionDiv" style="height: 500px">
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="panel-footer">

                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>



    let command_sequences_selected_ = -1
    let current_state_machine_ = 0
    let command_sequences_information_ = {}
    let gph_
    let command_sequences_information_visual_ = []
    let handlers_type_ = []
    let selected_handlers_information_ = []
    let current_handler_select_ = {}
    let index_command_sequences_information_visual_selected_ = -1;

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
            selectCommandSequenceState()
        }
        else if (new_state == 2)
        {
            createNewCommandSequenceState()
        }
        else if (new_state == 3)
        {
	        updateCommandSequenceState()
        }
        else{
            alertify.error("Error state")
            stateLogicMaching(0)
            return
        }
        current_state_machine_ = new_state
    }

    function idleState(){
        command_sequences_selected_ = -1;
        current_handler_select_ = {};
        index_command_sequences_information_visual_selected_ = -1;
        $("#commandSequenceDiv").hide();
        $("#createCommandSequenceButton").show();
        $("#modifyCommandSequenceButton").hide();
        $("#deleteCommandSequenceButton").hide();
        $("#listCommandAcctionToAddDiv").hide();
        $("#saveRunCancelButtonsDiv").hide();
        $("#commandParamsDiv").hide();
        $("#saveSequenceButton").hide();
        $("#cancelButton").hide();
        $("#runSequenceButton").hide();
        $("#addCommandInSequenceButton").hide();
        $("#commandSequenceNameInput").hide();
        $("#commandSequenceNameInput").val("");


        $("#commandButtonsDiv").hide();
        $("#playCommandButton").hide();
        $("#discardCommandInSequenceButton").hide();
        $("#commandSequenceLabel").hide();
        getCommandSequences();
        getHandlers();
    }

    function selectCommandSequenceState(){
        current_handler_select_ = {};
        index_command_sequences_information_visual_selected_ = -1;
        $("#createCommandSequenceButton").hide();
        $("#modifyCommandSequenceButton").show();
        $("#deleteCommandSequenceButton").show();
        $("#listCommandAcctionToAddDiv").hide();
        $("#commandSequenceLabel").show();
        $("#commandSequenceLabel").text(command_sequences_selected_);
        chargeCommandSequence(command_sequences_selected_);
        $("#commandSequenceDiv").show();
        $("#saveSequenceButton").hide();
        $("#discardSequenceButton").hide();
        $("#runSequenceButton").show();
        $("#commandParamsDiv").hide();
        $("#saveRunCancelButtonsDiv").show();
        $("#commandButtonsDiv").hide();
        $("#playCommandButton").hide();
        $("#discardCommandInSequenceButton").hide();
        $("#commandSequenceNameInput").hide();
        $("#commandSequenceNameInput").val("");

    }

    function createNewCommandSequenceState(){
        current_handler_select_ = {};
        index_command_sequences_information_visual_selected_ = -1;
        $("#saveRunCancelButtonsDiv").show();

        $("#createCommandSequenceButton").hide();
        $("#modifyCommandSequenceButton").hide();
        $("#deleteCommandSequenceButton").hide();
        $("#commandSequenceLabel").hide();
        $("#commandSequenceLabel").text("");
        initVisualization();
        chargeListOfHandlers();
        $("#commandSequenceDiv").show();

        $("#commandSequenceNameInput").show();
        $("#commandSequenceNameInput").val("");

        $("#listCommandAcctionToAddDiv").show();
        $("#saveSequenceButton").show();
        $("#discardSequenceButton").show();
        $("#runSequenceButton").hide();

    }

    function updateCommandSequenceState(){
        current_handler_select_ = {};
        index_command_sequences_information_visual_selected_ = -1;
        $("#saveRunCancelButtonsDiv").show();
        $("#createCommandSequenceButton").hide();
        $("#modifyCommandSequenceButton").hide();
        $("#deleteCommandSequenceButton").hide();
        $("#commandParamsDiv").hide();
        $("#commandSequenceNameInput").val($("#commandSequenceLabel").text());
        $("#commandSequenceLabel").show();
        chargeListOfHandlers();
        chargeCommandSequence(command_sequences_selected_);
        $("#commandSequenceDiv").show();
        $("#listCommandAcctionToAddDiv").show();
        $("#saveSequenceButton").show();
        $("#discardSequenceButton").show();
        $("#runSequenceButton").hide();
    }

    function chargeListOfHandlers(){
        var list_actions_element = document.getElementById("listActionsDiv");
        list_actions_element.innerHTML = "";
        jQuery.each(handlers_type_, function( handler_type_index, handler_type_foreach ){

            var button_element = document.createElement('button');
            button_element.id="action" + handler_type_foreach["name"] + "Button";
            button_element.onclick=function(){
                $("#removeActionButton").hide();
                $("#downActionButton").hide();
                $("#upActionButton").hide();
                current_handler_select_ = {};
                index_command_sequences_information_visual_selected_ = -1;
                chargeParamPanel(handler_type_foreach["name"], "Init", true);
            }
            button_element.className = "btn btn-block";
            button_element.textContent =  handler_type_foreach["name"];
            list_actions_element.appendChild(button_element);
        });
    }

    function deleteAction(){
        alertify.confirm('Delete action', 'Do you want delete the action?', function(){
            if(index_command_sequences_information_visual_selected_ == 0){
                alertify.error('Error to delete action');
                return;
            }
            //Update selected_handler_information:
            selected_handler_information_new = [];
            handler_new = {
                handler : "",
                previous_action: "",
                values: []
            }
            var i=1;
            for(i; i <= command_sequences_information_visual_["nodes"].length-1; i++){
                if(i < index_command_sequences_information_visual_selected_){
                    selected_handler_information_new.push(selected_handlers_information_[i-1]);
                }else if (i > index_command_sequences_information_visual_selected_){
                    var handler_new = Object.assign({}, selected_handlers_information_[i-1]);
                    if(i==2){
                        handler_new["previous_action"] = "Init";
                    }else{
                        handler_new["previous_action"] = selected_handler_information_new[selected_handler_information_new.length-1]["handler"];
                    }
                    selected_handler_information_new.push(handler_new)
                }
            }
            selected_handlers_information_ = selected_handler_information_new;
            updateVisualElements(selected_handlers_information_);
            $("#commandParamsDiv").hide();
            $("#listCommandAcctionToAddDiv").show();
            alertify.success('Deleted action');


        }, function(){alertify.success('Discarded')});
    }


    function moveUpAction(){
        if(index_command_sequences_information_visual_selected_ <= 1){
            alertify.error('Error to up action');
            return;
        }
        var index_handler_selected = index_command_sequences_information_visual_selected_ -1;
        //Update selected_handler_information:
        selected_handler_information_new = [];
        handler_new = {
            handler : "",
            previous_action: "",
            values: []
        }
        var i=0;
        for(i; i <= selected_handlers_information_.length-1; i++){
            if (i == (index_handler_selected - 1)){
                var handler_new = Object.assign({}, selected_handlers_information_[i+1]);
                if(i==0){
                    handler_new["previous_action"] = "Init";
                }else{
                    handler_new["previous_action"] = selected_handler_information_new[i-1]["handler"];
                }
                selected_handler_information_new.push(handler_new);
            }else if(i == index_handler_selected){
                var handler_new = Object.assign({}, selected_handlers_information_[i-1]);
                handler_new["previous_action"] = selected_handler_information_new[i-1]["handler"];
                selected_handler_information_new.push(handler_new);
            }else{
                var handler_new = Object.assign({}, selected_handlers_information_[i]);
                if(i!=0){
                    handler_new["previous_action"] = selected_handler_information_new[i-1]["handler"];
                }
                selected_handler_information_new.push(handler_new);
            }
        }
        selected_handlers_information_ = selected_handler_information_new;
        updateVisualElements(selected_handlers_information_);
        index_command_sequences_information_visual_selected_ --;
        $("#commandParamsDiv").hide();
        $("#listCommandAcctionToAddDiv").show();

    }

    function moveDownAction(){
        if(index_command_sequences_information_visual_selected_ >= command_sequences_information_visual_["nodes"].length -1){
            alertify.error('Error to down action');
            return;
        }
        var index_handler_selected = index_command_sequences_information_visual_selected_ -1;
        //Update selected_handler_information:
        selected_handler_information_new = [];
        handler_new = {
            handler : "",
            previous_action: "",
            values: []
        }
        var i=0;
        for(i; i <= selected_handlers_information_.length-1; i++){
            if (i == (index_handler_selected + 1)){

            }else if(i == index_handler_selected){

                var handler_new = Object.assign({}, selected_handlers_information_[i+1]);
                handler_new["previous_action"] = selected_handlers_information_[i]["handler"];
                if(i==0){
                    handler_new["previous_action"] = "Init";
                }else{
                    handler_new["previous_action"] = selected_handlers_information_[i]["handler"];
                }
                selected_handler_information_new.push(handler_new);

                handler_new = Object.assign({}, selected_handlers_information_[i]);
                handler_new["previous_action"] = selected_handler_information_new[i]["handler"];
                selected_handler_information_new.push(handler_new);

            }else{
                var handler_new = Object.assign({}, selected_handlers_information_[i]);
                if(i!=0){
                    handler_new["previous_action"] = selected_handler_information_new[i-1]["handler"];
                }
                selected_handler_information_new.push(handler_new);
            }
        }
        selected_handlers_information_ = selected_handler_information_new;
        updateVisualElements(selected_handlers_information_);
        index_command_sequences_information_visual_selected_ ++;
        $("#commandParamsDiv").hide();
        $("#listCommandAcctionToAddDiv").show();
    }

    function discardActionInSequence(){
        $("#commandParamsDiv").hide();
        $("#listCommandAcctionToAddDiv").show();
    }

    function addActionInSequence(){
        var command = "";
        $("#removeActionButton").hide();
        $("#downActionButton").hide();
        $("#upActionButton").hide();

        command = $("#paramsOfLabel").text();
        var i;
        handler = getTypesHandler(command);
        size_type_handler = handler.length;
        for(i=0; i<size_type_handler; i++){
            var element = $("#value_command_01_____"+i.toString());
            if(element.val() === ""){
                element.val("None");
            }
            if(handler[i].type=="bool"){
                command = command + " " +element[0].checked.toString();
            }else{
                command = command + " " +element.val();
            }
        }
        if(index_command_sequences_information_visual_selected_ <= 0){
            addElementInVisualization(command)
        }
        else
        {
            addElementInVisualization(command, index_command_sequences_information_visual_selected_-1)
        }
        $("#commandParamsDiv").hide();
        $("#listCommandAcctionToAddDiv").show();
    }

    function initVisualization(){
        if(gph_ != null){
            gph_.destroy();
        }
        container = document.getElementById("VisualizationActionDiv");

        command_sequences_information_visual_ = {
            nodes: [
                {
                    id: "Init_01_____0",
                    size: 100,
                    label: "Init",
                    color: "#0000FF",
                    shape: "cicle",
                    font: { face: "monospace", align: "left" }
                }
            ],
            edges: [

            ]
        }
        selected_handlers_information_ = [];
        gph_ = new vis.Network(container, command_sequences_information_visual_, options);
        gph_.on("click", function (params){onclickElementNetwork(params)});
    }

    function onclickElementNetwork(params)
    {
            console.log(params);
            index_command_sequences_information_visual_selected_ = parseInt(params["nodes"][0].split("_01_____")[1]);
            if(current_state_machine_ == 3 || current_state_machine_ == 2){
                if(index_command_sequences_information_visual_selected_ <= 1){
                    $("#upActionButton").hide();
                }else{
                    $("#upActionButton").show();
                }

                if(index_command_sequences_information_visual_selected_ >= command_sequences_information_visual_["nodes"].length -1 && index_command_sequences_information_visual_selected_!=0)
                {
                    $("#downActionButton").hide();
                }
                else
                {
                    $("#downActionButton").show();
                }
                if(index_command_sequences_information_visual_selected_ != 0){
                    $("#removeActionButton").show();
                }else{
                    $("#removeActionButton").hide();
                }
            }else{
                $("#removeActionButton").hide();
                $("#downActionButton").hide();
                $("#upActionButton").hide();
            }
            if(index_command_sequences_information_visual_selected_ == 0){
                if(current_state_machine_ == 3 || current_state_machine_ == 2){
                    $("#listCommandAcctionToAddDiv").show();
                }
                $("#commandParamsDiv").hide();
            }else{
                current_handler_select_ = selected_handlers_information_[index_command_sequences_information_visual_selected_-1];
                if(current_state_machine_==1){
                    $("#commandButtonsDiv").show();
                    $("#playCommandButton").show();
                    $("#discardCommandInSequenceButton").hide();
                    console.log(current_state_machine_);
                    chargeParamPanel(current_handler_select_["handler"], current_handler_select_["previous_action"] ,false, current_handler_select_["values"]);
                }else if(current_state_machine_ == 3 || current_state_machine_ == 2){
                    chargeParamPanel(current_handler_select_["handler"], current_handler_select_["previous_action"] ,true, current_handler_select_["values"]);
                }else{
                    chargeParamPanel(current_handler_select_["handler"], current_handler_select_["previous_action"] ,false, current_handler_select_["values"]);
                }
                $("#commandParamsDiv").show();
            }
    }

    var index_id = -1;

    function addElementInVisualization(command, element_id=-1){
        id_command = "";
        label_in_node = "";
        new_edge = {};
        types = [];
        handler_information = {
            handler: "",
            values: [],
            previous_action: ""
        }


        var command_split = command.split(" ");
        if(command_split.lenght < 1)
        {
            console.log("Error in the command: " + command + ". It is not correct.")
            return false;
        }
        var handler_command = command_split[0];

        if(element_id == -1) //New element
        {
            var last_handler_information_index = selected_handlers_information_.length - 1;
            if(last_handler_information_index == -1){
                handler_information["previous_action"] = "Init"
            }else{
                handler_information["previous_action"] = selected_handlers_information_[last_handler_information_index]["previous_action"];
            }
            handler_information["handler"] = handler_command;
            it_handler = 1
            for (it_handler; it_handler<command_split.length; it_handler++){
                handler_information["values"].push(command_split[it_handler]);
            }
            selected_handlers_information_.push(handler_information)
        }
        else //Update element
        {
            if(element_id > selected_handlers_information_.length - 1){
                console.log("Error to update command: " + command + ". The element_id " + element_id.toString() + " is bigger that " + (selected_handlers_information_.length - 1).toString());
                return false;
            }
            if(element_id == 0){
                selected_handlers_information_[element_id]["previous_action"] = "Init";
            }
            else
            {
                selected_handlers_information_[element_id]["previous_action"] = selected_handlers_information_[element_id-1]["handler"];
            }
            selected_handlers_information_[element_id]["handler"] =  handler_command;
            selected_handlers_information_[element_id]["values"] = [];
            it_handler = 1
            for (it_handler; it_handler<command_split.length; it_handler++){
                selected_handlers_information_[element_id]["values"].push(command_split[it_handler]);
            }
        }

        //Update visual schema using select_handlers_information_:
        return updateVisualElements (selected_handlers_information_);
    }



    function updateVisualElements (handlers_information){
        command_sequences_information_visual_ = {
            nodes: [
                {
                    id: "Init_01_____0",
                    size: 100,
                    label: "Init",
                    color: "#0000FF",
                    shape: "cicle",
                    font: { face: "monospace", align: "left" }
                }
            ],
            edges: [

            ]
        };
        it_select_handler_info = 0;
        for(it_select_handler_info; it_select_handler_info<handlers_information.length; it_select_handler_info++)
        {
            handler = handlers_information[it_select_handler_info]["handler"];
            types = getTypesHandler(handler);
            if(types == -1){
                console.error("The types to " + handler + " doesn't exist. I only show the corrects handler" );
                alertify.error("Error to charge the handler. Does the handler "+ handler +" exist?");
                gph_.setData(command_sequences_information_visual_);
                return false;
            }else if (types.length != handlers_information[it_select_handler_info]["values"].length)
            {
                console.error("The number of values in the handler " + handler + " ("+ it_select_handler_info +") , it is different to: " + types.length);
                console.error("Values in handler:")
                console.log(handlers_information[it_select_handler_info]["values"])
                console.error("Values in the type:")
                console.log(types)
                console.error("handlers_information:")
                console.log(handlers_information)
                alertify.error("Error to charge the handler. Did the handler "+ handler +" well formed?");
                gph_.setData(command_sequences_information_visual_);
                return false;
            }

            id_command = handler+"_01_____"+(it_select_handler_info+1).toString();
            new_edge = {
                from: command_sequences_information_visual_.nodes[it_select_handler_info]["id"],
                to: id_command,
                arrows: "to",
                physics: false,
                smooth: { type: "cubicBezier" }
            }

            label_in_node = handler + " \n\n ";
            var it = 0;
            for (it; it < handlers_information[it_select_handler_info]["values"].length; it++)
            {
                label_in_node = label_in_node + types[it]["description"]+ ": " + handlers_information[it_select_handler_info]["values"][it] + " \n ";
            }


            new_node = {
                id: id_command,
                size: 400,
                color: "#FFCFCF",
                shape: "box",
                font: { face: "monospace", align: "left" },
                label: label_in_node
            }
            command_sequences_information_visual_["nodes"].push(new_node);
            command_sequences_information_visual_["edges"].push(new_edge);

        }
        gph_.setData(command_sequences_information_visual_);
        return true;


    }

    function chargeCommandSequence(id){
        initVisualization();
        callService('<?php echo $robot_dictionary["service_command_sequencer_get_sequences"]?>', 'robot_simple_command_manager_msgs/GetCommandManagerList', {}, function (res) {
            if(res["success"] != true){
                alertify.error("Error to charge command sequences. Is it the command sequences running?");
                return;
            }
            else{
                jQuery.each(res["list"]["commands"], function( i, list ) {
                    if(list["id"]==id){
                        list["command"]=list["command"].replaceAll(" ; ", ";");
                        list["command"]=list["command"].replaceAll("; ", ";");
                        list["command"]=list["command"].replaceAll(" ;", ";");
                        list["command"].split(";").forEach(function(command_foreach)
                        {
                            if(!addElementInVisualization(command_foreach)){
                                deleteCommandSequence(id);
                            }
                        });
                    }
                });
            }
        });
    }

    function getCommandSequences(){
        callService('<?php echo $robot_dictionary["service_command_sequencer_get_sequences"]?>', 'robot_simple_command_manager_msgs/GetCommandManagerList', {}, function (res) {
            command_sequences_selected_ = -1;
            $('#sequenceCommandSelect').empty();
            $('#sequenceCommandSelect').append($('<option>', {
                value: -1,
                text: "--Choose command--"
                }));
            $('[name=options]').val(-1);
            if(res.success != true){
                alertify.error("Error to charge command sequences. Is it the command sequences running?");
                return;
            }
            else{
                jQuery.each(res["list"]["commands"], function( i, val ) {
                    $('#sequenceCommandSelect').append($('<option>', {
                        value: val["id"],
                        text: val["id"]
                    }));
                });
            }
        });
    }

    $("#sequenceCommandSelect" ).change(function () {
        change_command_sequence_selected = false
        $( "#sequenceCommandSelect option:selected" ).each(function() {
            change_command_sequence_selected = ( $( this ).val() != command_sequences_selected_)
            command_sequences_selected_ = $( this ).val();
        });
        if(!change_command_sequence_selected){
	    return;
        }
        if(command_sequences_selected_ == -1){
            stateLogicMaching(0);
        }else{
            stateLogicMaching(1);
        }
    }).change();

    function getHandlers(){
        //handlers_type_ =[]
        callService('<?php echo $robot_dictionary["service_command_manager_get_handlers"]?>', 'robot_simple_command_manager_msgs/GetHandlers', {}, function (res) {
            handlers_type_ =[];
            console.log(res);
            jQuery.each(res["handlers"], function( handler_name_foreach_index, handler_name_foreach ){
                srv = {
                    name: handler_name_foreach
                }
                callService('<?php echo $robot_dictionary["service_command_manager_get_handler_info"]?>', 'robot_simple_command_manager_msgs/GetHandlerInfo', srv, function (get_handler_info_res) {
                    if(!get_handler_info_res["success"]){
                        alertify.error("Error to get handlers info");
                        return;
                    }else if(get_handler_info_res["description"].length!=get_handler_info_res["types"].length){
                        alertify.error("Error to charge handler " + handler_name_foreach + ". The number of types and descriptions are different.");
                        return;
                    }

                    handler = {
                        name: handler_name_foreach,
                        values: []
                    }
                    jQuery.each(get_handler_info_res["description"], function( handler_name_foreach_index, handler_name_foreach ){
                        values = {
                            description: get_handler_info_res["description"][handler_name_foreach_index],
                            type: get_handler_info_res["types"][handler_name_foreach_index]
                        }
                        handler["values"].push(values);
                    });
                    handlers_type_.push(handler);
                });
            });
        });
    }

    function getTypesHandler(handler){
        for(i = 0; i < handlers_type_.length; i++){
            if(handler == handlers_type_[i]["name"]){
                return handlers_type_[i]["values"];
            }
        }
        alertify.error("The handler " + handler + " doesn't exist");
        return -1;
    }

    let current_environment = ""

    subscribe('<?php echo $robot_dictionary["topic_robot_local_state"];?>', 'robot_local_control_msgs/Status', function (message) {
        if(message.localization_status.state == "READY")
        {
            current_environment = message.localization_status.environment;
        }
        else
        {
            current_environment = "";
        }

    });

    var temp_value;

    function chargeTagList(id_elem, value){
        if(current_environment == ""){

        }else{
            temp_value = value;
            srv = {
                environment: current_environment
            }
            callService('<?php echo $robot_dictionary["service_poi_manager_get_poi_list"]?>', 'poi_manager_msgs/GetPOIs', srv, function (res) {
                $('#'+id_elem).empty();
                $('#'+id_elem).append($('<option>', {
                    value: "",
                    text: "--Choose--"
                }));
                res["p_list"].forEach(function(val) {
                    $('#'+id_elem).append($('<option>', {
                        value: val.name,
                        text: val.name
                    }));
                })
                $('#'+id_elem).val(temp_value);
            });
        }
    }

    function chargeParamPanel(type_of_command, previous_action="Init", edit=false, values=[]){
        $("#listCommandAcctionToAddDiv").hide();
        $("#playCommandButton").show();
        $("#paramsOfLabel").text(type_of_command);
        if(edit){
            $("#addCommandInSequenceButton").show();
            $("#discardCommandInSequenceButton").show();
            previous_action_id = command_sequences_information_visual_["nodes"][command_sequences_information_visual_.nodes.length-1]["id"].split("_01_____")[0];
            $("#previousActionLabel").text(previous_action_id);
        }else{
            $("#addCommandInSequenceButton").hide();
            $("#discardCommandInSequenceButton").hide();
            $("#previousActionLabel").text(previous_action);
        }
        handler = getTypesHandler(type_of_command)
        if(values.length != handler.length){
            values=[];
            jQuery.each(handler, function( handler_type_index, handler_type_foreach ){
                if(type_of_command.includes("GOTO_TAG") && handler_type_foreach["description"].includes("environment")){
                    values.push(current_environment);
                }
                else if(handler_type_foreach["type"]=="float"){
                    values.push(0.0);
                }else if(handler_type_foreach["type"]=="integer"){
                    values.push(0);
                }else if(handler_type_foreach["type"]=="bool"){
                    values.push(false);
                }else if(handler_type_foreach["type"]=="str"){
                    values.push("");
                }else{
                    values.push("");
                }
            });
        }
        panel_commands = document.getElementById("panelCommands")
        panel_commands.innerHTML = "";
        var element;
        var i;
        for(i=0; i < handler.length; i++){
            if(type_of_command.includes("GOTO_TAG") && handler[i]["description"].includes("tag")){
                element = document.getElementById("ListDiv").cloneNode(true);
                element.id = "elem_01_____" + i;
                element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                element.children[0].children[0].textContent=handler[i]["description"]+":";
                element.children[0].children[1].id = "value_command_01_____"+i;
                chargeTagList(element.children[0].children[1].id, values[i]);
                //console.log(values[i]);
                //element.children[0].children[1].value=values[i];
                element.children[0].children[1].disabled=!edit;
            }else if(type_of_command.includes("GOTO_TAG") && handler[i]["description"].includes("environment")){
                element = document.getElementById("StringDiv").cloneNode(true);
                element.id = "elem_01_____" + i;
                element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                element.children[0].children[0].textContent=handler[i]["description"]+":";
                element.children[0].children[1].id = "value_command_01_____"+i;
                element.children[0].children[1].value=values[i];
                element.children[0].children[1].readOnly=!edit;
            }else{
                if(handler[i]["type"]=="float"){
                    element = document.getElementById("DoubleNumberDiv").cloneNode(true);
                    element.id = "elem_01_____" + i;
                    element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                    element.children[0].children[0].textContent=handler[i]["description"]+":";
                    element.children[0].children[1].id = "value_command_01_____"+i;
                    element.children[0].children[1].value=values[i];
                    element.children[0].children[1].readOnly=!edit;
                }else if(handler[i]["type"]=="integer"){
                    element = document.getElementById("IntegerNumberDiv").cloneNode(true);
                    element.id = "elem_01_____" + i;
                    element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                    element.children[0].children[0].textContent=handler[i]["description"]+":";
                    element.children[0].children[1].id = "value_command_01_____"+i;
                    element.children[0].children[1].value=values[i];
                    element.children[0].children[1].readOnly=!edit;
                }else if(handler[i]["type"]=="bool"){
                    element = document.getElementById("BooleanDiv").cloneNode(true);
                    /*
                    element.children[0].textContent = handler[i]["description"];
                    element.children[0].children[0].id = "value_command_01_____"+i;
                    element.children[0].children[0].checked = values[i];
                    element.children[0].children[0].readOnly=!edit;
                    */
                    element.id = "elem_01_____" + i;
                    element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                    element.children[0].children[0].textContent=handler[i]["description"]+":";
                    element.children[0].children[1].id = "value_command_01_____"+i;
                    element.children[0].children[1].checked =(values[i]=="true");
                    element.children[0].children[1].disabled=!edit;

                }else if(handler[i]["type"]=="str"){
                    element = document.getElementById("StringDiv").cloneNode(true);
                    element.id = "elem_01_____" + i;
                    element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                    element.children[0].children[0].textContent=handler[i]["description"]+":";
                    element.children[0].children[1].id = "value_command_01_____"+i;
                    element.children[0].children[1].value=values[i];
                    element.children[0].children[1].readOnly=!edit;
                }else{
                    element = document.getElementById("StringDiv").cloneNode(true);
                    element.id = "elem_01_____" + i;
                    element.children[0].children[0].setAttribute("for", "value_command_01_____"+i);
                    element.children[0].children[0].textContent=handler[i]["description"]+":";
                    element.children[0].children[1].id = "value_command_01_____"+i;
                    element.children[0].children[1].value=values[i];
                    element.children[0].children[1].readOnly=!edit;
                }
            }
            panel_commands.appendChild(element);
        }
        $("#commandButtonsDiv").show();
        $("#commandParamsDiv").show();
    }

    function playCommand(){
        alertify.confirm('Play command', 'Do you want start the command?', function(){executeCurrentCommand() }, function(){alertify.success('Discarted')});
    }

    function executeCurrentCommand(){
        command = "";
        if(current_handler_select_["handler"] != undefined){
            command = command + current_handler_select_["handler"] + " ";
            jQuery.each(current_handler_select_["values"], function (index_value_foreach, value_foreach)
            {
                console.log("-->" + command)
                command = command + value_foreach + " ";
            });
        }else{
            command = command = $("#paramsOfLabel").text() + " ";
            handler = getTypesHandler($("#paramsOfLabel").text());
            size_type_handler = handler.length;
            for(i=0; i<size_type_handler; i++){
                var element = $("#value_command_01_____"+i.toString());
		if(element.val() === ""){
			element.val("None");
		}
                if(handler[i].type=="bool"){
                    command = command + " " +element[0].checked.toString();
                }else{
                    command = command + " " +element.val();
                }
            }

        }

        srv_command = {
            command: command
        }
        console.log(srv_command);
        callService('<?php echo $robot_dictionary["service_command_manager_command"]?>', 'robot_simple_command_manager_msgs/SetCommandString', srv_command, function (res) {
            if(res["ret"]["success"]){
                alertify.success('Exectuting command');
            }else{
                alertify.error(res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
            }

        });
    }

    function runSequence(){
        alertify.confirm('Run sequence', 'Do you want run the sequence?', function(){executeSequence() }, function(){alertify.success('Discarded')});
    }

    function getSequenceCommand(){
        var command = "";
        var sequence = command_sequences_information_visual_["nodes"];
        jQuery.each(sequence, function(sequence_index, sequence_foreach ){
            if(sequence_index!=0){
                var type_command = sequence_foreach["id"].split("_01_____")[0];
                command = command + type_command;
                var values = sequence_foreach["label"];
                values = values.replaceAll(": ", ":");
                values = values.split("\n\n")[1];
                values = values.split(" \n ");
                var i;
                for(i=0; i<=values.length-2; i++){
                    command = command + " " +values[i].split(":")[1];
                }
                command = command + " ; ";

            }
        });
        return command.slice(0, -3);
    }

    function deleteSequence(){
        alertify.confirm('Delete sequence', 'Do you want delete the sequence?', function(){deleteCurrentSequence() }, function(){});
    }

    function deleteCurrentSequence(){
        if(current_state_machine_==1){
            var delete_sequence_srv = {
                Command: {
                    id: $("#commandSequenceLabel").text(),
                    command: ""
                }
            };

            callService('<?php echo $robot_dictionary["service_command_sequencer_remove_sequence"]?>', 'robot_simple_command_manager_msgs/ManageCommandManager', delete_sequence_srv, function (res) {
                if(res["ret"]["success"]){
                    alertify.success('Removed: ' + $("#commandSequenceNameInput").val());
                    stateLogicMaching(0);
                }else{
                    alertify.error("Error, to save sequence. Message: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
                }
            });
        }
    }

    function saveSequence(){
        if($("#commandSequenceNameInput").val() == ""){
            alertify.error("It is necessary to give a name to the sequence");
            return;
        }
        var command = getSequenceCommand();
        if(command_sequences_information_visual_["nodes"].lenght == 1 || command == ""){
            alertify.error("It is necessary to add at least one action");
            return;
        }
        if(current_state_machine_==2 || current_state_machine_==3){
            var create_sequence_srv = {
                Command: {
                    id: $("#commandSequenceNameInput").val(),
                    command: command
                }
            };
            console.log("Send to "+ '<?php echo $robot_dictionary["service_command_sequencer_add_sequence"]?>' + ", " + create_sequence_srv);
            callService('<?php echo $robot_dictionary["service_command_sequencer_add_sequence"]?>', 'robot_simple_command_manager_msgs/ManageCommandManager', create_sequence_srv, function (res) {
                console.log('Response')
                if(res["ret"]["success"]){
                    if(current_state_machine_==2){
                        alertify.success('Saved: ' + $("#commandSequenceNameInput").val());
                    }else{
                        alertify.success('Updated: ' + $("#commandSequenceNameInput").val());
                    }
                    stateLogicMaching(0);
                }else{
                    alertify.error("Error, to save sequence. Message: " + res["ret"]["message"] + ". Code: " + res["ret"]["code"]);
                }
            });
        }else{
            alertify.error("Error, current state machine is not exist");
        }
    }

    function executeSequence(){
        if(current_state_machine_ == 1){
            command = $("#commandSequenceLabel").text();
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
        }else{
            alertify.error("Error, current state machine is not exist");
        }
    }

    function cancelSequence(){
        srv = {};
        callService('<?php echo $robot_dictionary["service_command_sequencer_cancel"]?>', 'std_srvs/Trigger', srv, function (res) {
            if(res["success"]){
                alertify.success('Cancelled mission');
            }else{
                alertify.error("Error to cancel mission. Error: " + res["message"]);
                return;
            }
        });
    }

    function deleteCommandSequence(id_command_sequence){

    }

    function filterFunction() {
        var input, filter, ul, li, a, i;
        input = document.getElementById("searchActionInput");
        filter = input.value.toUpperCase();
        div = document.getElementById("listActionsDiv");
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

    function isNumber(evt) {
        evt = (evt) ? evt : window.event;
        var charCode = (evt.which) ? evt.which : evt.keyCode;
        if ( (charCode > 31 && charCode < 48) || charCode > 57 && charCode != 45) {
            return false;
        }
        return true;
    }

    function isFloat(evt) {
        evt = (evt) ? evt : window.event;
        var charCode = (evt.which) ? evt.which : evt.keyCode;
        if (charCode == 46 || charCode == 45){
            return true;
        }
        if ((charCode > 31 && charCode < 48) || charCode > 57 || charCode == 45) {
            return false;
        }
        return true;
    }

    $(document).ready(function () {
    	stateLogicMaching(0);
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
            $("#executinCollapse").show();
        }else{

            $("#executingDiv").hide();
            $("#executinCollapse").hide();

        }
        $("#sequenceErrorLabel").text(message.last_error);
        if(message.last_error==""){
            $("#sequenceStateErrorDiv").hide();
            $("#errorPanelDiv").hide()
            $("#executinErrorCollapse").hide()
        }else{
            $("#errorPanelDiv").show()
            $("#executinErrorCollapse").show();
            $("#sequenceStateErrorDiv").show();

        }
    });

</script>
