<div class="backgroundPageWeb" style="margin-top:60px">
    <div class="container-fluid">
    
        <div class="row">
            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                <div id="menu" class="float-right">
                    <span id="menu-navi">
                        <button type="button" class="btn btn-default btn-sm move-day" onclick="previousWeek()">
                            <i class="glyphicon glyphicon-arrow-left" data-action="move-prev"></i>
                        </button>
                        <button type="button" class="btn btn-default btn-sm move-today" onclick="currentWeek()">
                            <i class="glyphicon glyphicon-calendar"></i>
                        </button>
                        <button type="button" class="btn btn-default btn-sm move-day" onclick="nextWeek()">
                            <i class="glyphicon glyphicon-arrow-right" data-action="move-next"></i>
                        </button>
                        <button type="button" class="btn btn-default btn-sm" data-toggle="modal" data-target="#myModal" onclick="changeModalState(0)">
                            <span class="glyphicon glyphicon-plus"></span> New event
                        </button>
                        <h1><label id="monthLabel" style="font-size:30px !important"></label></h1>
                        <hr>
                    </span>
                </div>
                <div id="calendar"></div>
            </div>
            
        </div>

        <!-- Modal -->

        <div id="myModal" class="modal fade" role="dialog" data-keyboard="false" data-backdrop="static">
        <div class="modal-dialog">

            <!-- Modal content-->
            <div class="modal-content">
            <div class="modal-header">
                <button type="button" class="close" data-dismiss="modal">&times;</button>
                <h4 id="modalTitle" class="modal-title">New event
                    
                </h4>
                <button id="editEventButton" type="button" class="btn btn-default btn-sm" onclick="changeModalState(2)">
                    <span class="glyphicon glyphicon-pencil"></span> Edit 
                </button>
                <button id="removeEventButton" type="button" class="btn btn-default btn-sm" onclick="removeEvent()">
                    <span class="glyphicon glyphicon-trash"></span> Remove 
                </button>
            </div>
            <div class="modal-body">
                <div class="row">
                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                        <div class="form-group">
                            <label for="titleInput">Title:</label>
                            <input type="text" class="form-control" id="titleInput">
                        </div>
                    </div>
                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                        <div class="form-group">
                            <label for="missionSelectInput">Mission:</label>
                            <select class="form-control" id="missionSelectInput">
                                <option>-- Choose mission --</option>
                            </select>
                        </div>
                    </div>
                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="dateDiv">
                        <div class="row">
                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="startDateCbDiv">
                                <div class="form-group">
                                    <input class="form-check-input" type="checkbox" value="" id="startDateCb">
                                    <label class="form-check-label" for="startDateCb">
                                        Define start date
                                    </label>
                                </div>
                            </div>
                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="startDateInputDiv">
                                <div class="form-group">
                                    <label for="startDateInput">Start date:</label>
                                    <input id="startDateInput" type="date"/>
                                </div>
                            </div>                        
                        </div>
                        <div class="row">
                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="endDateCbDiv">
                                <div class="form-group">
                                    <input class="form-check-input" type="checkbox" value="" id="endDateCb">
                                    <label class="form-check-label" for="endDateCb">
                                        Define end date
                                    </label>
                                </div>
                            </div>
                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="endDateInputDiv">
                                <div class="form-group">
                                    <label for="endDateInput">End date:</label>
                                    <input id="endDateInput" type="date"  data-date-format="yyyy/mm/dd"/>
                                </div>
                            </div> 
                        </div>
                    </div>
                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="startDiv">
                        <div class="form-group">
                            <label for="timeInput">Time:</label>
                            <input id="timeInput" type="time"/>
                        </div>
                    </div>
                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="repeatWeekDayDiv">
                        <div class="row">
                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                <label>Week repeate:</label>
                            </div>
                            <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                                <label class="checkbox-inline"><input type="checkbox" name="repeatWeekDayCB" id="weekDay0" value="0">Monday</label>
                            </div>
                            <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                                <label class="checkbox-inline"><input type="checkbox" name="repeatWeekDayCB" id="weekDay1" value="1">Tuesday</label>
                            </div>
                            <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                                <label class="checkbox-inline"><input type="checkbox" name="repeatWeekDayCB" id="weekDay2" value="2">Wednesday</label>
                            </div>
                            <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                                <label class="checkbox-inline"><input type="checkbox" name="repeatWeekDayCB" id="weekDay3" value="3">Thursday</label>
                            </div>
                            <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                                <label class="checkbox-inline"><input type="checkbox" name="repeatWeekDayCB" id="weekDay4" value="4">Friday</label>
                            </div>
                            <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                                <label class="checkbox-inline"><input type="checkbox" name="repeatWeekDayCB" id="weekDay5" value="5">Saturday</label>
                            </div>
                            <div class="col-md-4 col-xl-4 text-center">
                            </div>
                            <div class="col-xs-6 col-ms-6 col-md-4 col-xl-4">
                                <label class="checkbox-inline"><input type="checkbox" name="repeatWeekDayCB" id="weekDay6" value="6">Sunday</label>
                            </div>
                        </div>
                        <div class="row" id="chooseAllDaysDiv">
                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12 text-right">
                                <button class="btn btn-default btn-sm" onclick="chooseAllDays()">Choose all days</button>
                            </div>
                            
                        </div>
                        
                    </div>
                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="repeatDiv">
                        <div class="row">
                            <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12">
                                <label class="checkbox-inline"><input type="checkbox" id="repeatCBInput">Repeat</label>
                                <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="repeatTimeSlotDiv">
                                    <div class="form-group">
                                        <label for="timeSlotInput">Time slot (in hours):</label>
                                        <input type="number" class="form-control" id="timeSlotInput" min=0 value="1"/>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-xs-12 col-ms-12 col-md-12 col-xl-12" id="enableDiv">
                        <label class="checkbox-inline"><input type="checkbox" id="enableCBInput">Enable</label>
                    </div>
                    
                </div>
            </div>
            <div class="modal-footer">
                <div class="row">
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-left">
                        <button type="button" class="btn btn-default" id="SaveEventButton" onclick="saveEvent()">Save</button>
                    </div>
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-center" >
                        <!--button type="button" class="btn btn-default" data-dismiss="modal">Save</button-->
                    </div>
                    <div class="col-xs-4 col-ms-4 col-md-4 col-xl-4 text-right">
                        <button type="button" class="btn btn-default" data-dismiss="modal" onclick="$('#myModal').modal('hide');">Close</button>
                    </div>
                </div>
                
            </div>
        </div>
    </div>

    
</div>

<script>    
    var events_ = null;
    var calendar_structure_ = [];
    var modal_state_ = 0;
    var last_charge_event_ = null;

    const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];


    $( document ).ready(function() {
        document.getElementById("repeatCBInput").addEventListener('change', (event) => {
            if (event.currentTarget.checked) {
                $("#repeatTimeSlotDiv").show();
            } else {
                $("#repeatTimeSlotDiv").hide();
            }
        });
        

    document.getElementById("startDateCb").addEventListener('change', (event) => {
        if (event.currentTarget.checked) {
            $("#startDateInputDiv").show();
        } else {
            $("#startDateInputDiv").hide();
        }
    });

    document.getElementById("endDateCb").addEventListener('change', (event) => {
        if (event.currentTarget.checked) {
            $("#endDateInputDiv").show();
            
        } else {
            $("#endDateInputDiv").hide();
        }
    });


        $("#monthLabel").text(monthNames[calendar.getDate().getMonth()] + " " + calendar.getDate().getFullYear());
        chargeMissionsList();
        chargeEvents();
    });

    function changeModalState(new_modal_state, charge_event=null) {
        if(new_modal_state == 0) //New event
        {
            chargeMissionsList();
            $("#modalTitle").text("New event");
            $("#editEventButton").hide();
            $("#removeEventButton").hide();
            $("#titleInput").prop('disabled', false);
            $("#titleInput").val("");
            $("#titleInput").show();
            $("#missionSelectInput").val(-1);
            $("#missionSelectInput").change();
            $("#missionSelectInput").prop('disabled', false);
            var date = new Date();
            var currentTime = date.toISOString().substring(11,16);
            $("#timeInput").val(currentTime);
            $("#timeInput").prop('disabled', false);
            $("#enableCBInput").prop("checked", true);
            $("#enableCBInput").prop('disabled', false);
            $("#repeatCBInput").prop("checked", false);
            $("#repeatCBInput").prop('disabled', false);
            $("#repeatTimeSlotDiv").hide();
            $("input[type='checkbox'][name='repeatWeekDayCB']").prop("checked", false);
            $("input[type='checkbox'][name='repeatWeekDayCB']").prop("disabled", false);
            $("#weekDayAll").prop("checked", true);
            $("#timeSlotInput").val("1");
            $("#timeSlotInput").prop("disabled", false);
            var day = ("0" + date.getDate()).slice(-2);
            var month = ("0" + (date.getMonth() + 1)).slice(-2);
            var today = date.getFullYear()+"-"+ month +"-"+(day);
            $('#startDateInput').val(today);
            $('#endDateInput').val(today);
            $("#startDateCb").prop("checked", true);
            $("#startDateCb").prop('disabled', false);
            $('#startDateCb').prop("disabled",false);
            $('#endDateInput').prop("disabled",false);
            $("#startDateInput").prop("disabled",false);
            $("#endDateInput").prop("disabled",false);
            $("#startDateInputDiv").show();
            $("#endDateCb").prop("checked", false);
            $("#endDateCb").prop('disabled', false);
            $("#endDateInputDiv").hide();
            $("#chooseAllDaysDiv").show();

            $("#SaveEventButton").show();
        }
        else if(new_modal_state == 1){ //Show event
            $("#modalTitle").text("Event");
            $("#editEventButton").show();
            $("#removeEventButton").hide();
            $("#titleInput").prop('disabled', true);
            $("#titleInput").val("");
            $("#titleInput").show();
            $("#missionSelectInput").val(-1);
            $("#missionSelectInput").change();
            $("#missionSelectInput").prop('disabled', true);
            var date = new Date();
            var currentTime = date.toISOString().substring(11,16);
            $("#timeInput").val(currentTime);
            $("#timeInput").prop('disabled', true);
            $("#enableCBInput").prop("checked", true);
            $("#enableCBInput").prop('disabled', true);
            $("#repeatCBInput").prop("checked", false);
            $("#repeatCBInput").prop('disabled', true);
            $('#startDateCb').prop("disabled",true);
            $('#endDateInput').prop("disabled",false);
            $("#startDateInput").prop("disabled",true)
            $("#endDateInput").prop("disabled",true)        
            $("#repeatTimeSlotDiv").hide();
            $("input[type='checkbox'][name='repeatWeekDayCB']").prop("checked", false);
            $("input[type='checkbox'][name='repeatWeekDayCB']").prop("disabled", true);
            $("#weekDayAll").prop("checked", true);
            $("#timeSlotInput").val("1");
            $("#timeSlotInput").prop("disabled", true);
            $("#startDateCb").prop('disabled', true);
            $("#endDateCb").prop("disabled", true);
            $("#chooseAllDaysDiv").hide();
            $("#SaveEventButton").hide();
            chargeEvent(charge_event);
            last_charge_event_ = charge_event;
        }
        else if(new_modal_state == 2){ //Edit event
            $("#modalTitle").text("Edit event");
            $("#editEventButton").hide();
            $("#removeEventButton").show();
            $("#titleInput").prop('disabled', true);
            $("#titleInput").show();
            $("#missionSelectInput").prop('disabled', false);
            $("#timeInput").prop('disabled', false);
            $("#enableCBInput").prop('disabled', false);
            $("#repeatCBInput").prop('disabled', false);
            $("input[type='checkbox'][name='repeatWeekDayCB']").prop("disabled", false);
            $("#timeSlotInput").prop("disabled", false);
            $("#startDateCb").prop('disabled', false);
            $("#endDateCb").prop("disabled", false);
            $("#startDateInput").prop("disabled",false);
            $("#endDateInput").prop("disabled",false);
            $("#chooseAllDaysDiv").show();
            $("#SaveEventButton").show();
        }
        modal_state_ = new_modal_state;

    }

    function chooseAllDays()
    {
        var checkboxes = document.querySelectorAll('input[type=checkbox][name="repeatWeekDayCB"]')
        for (var i = 0; i < checkboxes.length; i++) {
            checkboxes[i].checked = true
        }

    }

    function chargeEvent(event){
        $("#titleInput").val(event["id"]);
        var checkboxes = document.querySelectorAll('input[type=checkbox][name="repeatWeekDayCB"]')
        for(var i=0; i<checkboxes.length; i++){
            if(event["wk"].includes(parseInt(checkboxes[i].value)))
            {
                checkboxes[i].checked = true;
            }
            else
            {
                checkboxes[i].checked = false;
            }
            
        }
        /*
        for(var i=0; i<event["wk"].length; i++){

            checkboxes[i].checked = true;
        }
        */
        /*
        if(split[0] == "A"){
            $("#weekDayAll").prop("checked", true);
        }else{
            $("#weekDay"+split[0]).prop("checked", true);
        }*/
        if(event["date_start"]!="")
        {
            event["date_start"] = event["date_start"].replaceAll("/","-");
            start_date_year = event["date_start"].split("-")[0];
            start_date_month = ("0" + event["date_start"].split("-")[1]).slice(-2);
            start_date_day = ("0" + event["date_start"].split("-")[2]).slice(-2);
            
            $("#startDateInput").val(start_date_year + "-" + start_date_month + "-" + start_date_day);
            $("#startDateInputDiv").show();
            $("#startDateCb").prop("checked", true);
        }else{
            $("#startDateInputDiv").hide();
            $("#startDateCb").prop("checked", false);
        }
        if(event["date_end"]!="")
        {
            event["date_end"] = event["date_end"].replaceAll("/","-");
            end_date_year = event["date_end"].split("-")[0];
            end_date_month = ("0" + event["date_end"].split("-")[1]).slice(-2);
            end_date_day = ("0" + event["date_end"].split("-")[2]).slice(-2);
            $("#endDateInput").val(end_date_year + "-" + end_date_month + "-" + end_date_day);
            $("#endDateInputDiv").show();
            $("#endDateCb").prop("checked", true);
        }else{
            $("#endDateInputDiv").hide();
            $("#startDateCb").prop("checked", false);
        }
        event["hour"] = ("0"+ event["hour"]).slice(-2);
        event["minute"] = ("0"+ event["minute"]).slice(-2);


        $("#timeInput").val(event["hour"]+":"+event["minute"]);
        var repeat = event["repeat_hour"];
        if(repeat == "0"){
            $("#repeatCBInput").prop("checked", false);
            $("#repeatTimeSlotDiv").hide();
            $("#timeSlotInput").val("1");
        }else{
            $("#repeatCBInput").prop("checked", true);
            $("#repeatTimeSlotDiv").show();
            $("#timeSlotInput").val(repeat);
        }
        if(event["enabled"]){
            $("#enableCBInput").prop('checked', true);
        }else{
            $("#enableCBInput").prop('checked', false);
        }
        $("#missionSelectInput").val(event["commands"]).change();
        //$("#missionSelectInput").change();
    }

    function saveEvent(){
        if($("#titleInput").val() == ""){
            alertify.error("It is necessary to name the event.");
            return;
        }

        if($("#missionSelectInput").val() == -1){
            alertify.error("It is necessary select a mission.");
            return;
        }
        var checkboxes = document.querySelectorAll('input[type=checkbox][name="repeatWeekDayCB"]:checked')
        if(checkboxes.lenght <= 0){
            alertify.error("It is necessary select at least one day of week.");
            return;
        }

        var wk = []
        

        for (var i = 0; i < checkboxes.length; i++) {
            wk.push(parseInt(checkboxes[i].value))
        }
        
        start_date = ""
        end_date = ""
        if($("#startDateCb").prop("checked")){
            start_date = new Date($("#startDateInput").val())
            start_date = start_date.getFullYear() + "/" + (start_date.getMonth() + 1) + "/" + start_date.getDate()
        }
        if($("#endDateCb").prop("checked")){
            end_date = new Date($("#endDateInput").val())
            end_date = end_date.getFullYear() + "/" + (end_date.getMonth() + 1) + "/" + end_date.getDate()
        }
        repeat = 0;
        if($("#repeatCBInput").prop("checked"))
        {
            repeat = parseInt($("#timeSlotInput").val());
        }
        srv = {
            event: 
            {
                id:$("#titleInput").val(),
                wk: wk,
                date_start: start_date,
                date_end: end_date,
                hour: parseInt($("#timeInput").val().split(":")[0]),
                minute: parseInt($("#timeInput").val().split(":")[1]),
                repeat_hour: repeat,
                enabled: $("#enableCBInput").prop("checked"),
                commands: $("#missionSelectInput").val()
            }
        }

        callService('<?php echo $robot_dictionary["service_add_scheduler"]?>', 'robot_simple_command_manager_msgs/ManageCommandManager', srv, function (res) {
            if(res["ret"]["success"]){
                if(modal_state_ == 0){
                    alertify.success("Added new event");
                }else{
                    alertify.success("Updated new event");
                }
                chargeEvents();
                $('#myModal').modal('hide');
            }else{
                alertify.error(res["ret"]["message"]);
            }
        });
        
    }
    
    var calendar = new tui.Calendar(document.getElementById('calendar'), {
    defaultView: 'month',
    taskView: ['milestone', 'task'],    // Can be also ['milestone', 'task']
    scheduleView: ['allday', 'time'],  // Can be also ['allday', 'time']
    useCreationPopup: false,
    template: 
    {
        milestone: function(schedule) {
            return '<span class="calendar-font-icon ic-milestone-b"></span> <span style="background-color: ' + schedule.bgColor + '">' + schedule.title + '</span>';
        },
        milestoneTitle: function() {
            return '<span class="tui-full-calendar-left-content">MILESTONE</span>';
        },
        task: function(schedule) {
            return '#' + schedule.title;
        },
        taskTitle: function() {
            return '<span class="tui-full-calendar-left-content">TASK</span>';
        },
        allday: function(schedule) {
            return getTimeTemplate(schedule, true);
        },
        alldayTitle: function() {
            return '<span class="tui-full-calendar-left-content">ALL DAY</span>';
        },
        time: function(schedule) {
            return '<strong>' + moment(schedule.start.getTime()).format('HH:mm') + '</strong> ' + schedule.title;
        },
        goingDuration: function(schedule) {
            return '<span class="calendar-icon ic-travel-time"></span>' + schedule.goingDuration + 'min.';
        },
        comingDuration: function(schedule) {
            return '<span class="calendar-icon ic-travel-time"></span>' + schedule.comingDuration + 'min.';
        },
        monthMoreTitleDate: function(date, dayname) {
            var day = date.split('.')[2];

            return '<span class="tui-full-calendar-month-more-title-day">' + day + '</span> <span class="tui-full-calendar-month-more-title-day-label">' + dayname + '</span>';
        },
        monthMoreClose: function() {
            return '<span class="tui-full-calendar-icon tui-full-calendar-ic-close"></span>';
        },
        monthGridHeader: function(dayModel) {
            var date = parseInt(dayModel.date.split('-')[2], 10);
            var classNames = ['tui-full-calendar-weekday-grid-date '];

            if (dayModel.isToday) {
                classNames.push('tui-full-calendar-weekday-grid-date-decorator');
            }

            return '<span class="' + classNames.join(' ') + '">' + date + '</span>';
        },
        monthGridHeaderExceed: function(hiddenSchedules) {
            return '<span class="weekday-grid-more-schedules">+' + hiddenSchedules + '</span>';
        },
        monthGridFooter: function() {
            return '';
        },
        monthGridFooterExceed: function(hiddenSchedules) {
            return '';
        },
        monthDayname: function(model) {
            return (model.label).toString().toLocaleUpperCase();
        },
        weekDayname: function(model) {
            return '<span class="tui-full-calendar-dayname-date">' + model.date + '</span>&nbsp;&nbsp;<span class="tui-full-calendar-dayname-name">' + model.dayName + '</span>';
        },
        weekGridFooterExceed: function(hiddenSchedules) {
            return '+' + hiddenSchedules;
        },
        dayGridTitle: function(viewName) {

            // use another functions instead of 'dayGridTitle'
            // milestoneTitle: function() {...}
            // taskTitle: function() {...}
            // alldayTitle: function() {...}

            var title = '';
            switch(viewName) {
                case 'milestone':
                    title = '<span class="tui-full-calendar-left-content">MILESTONE</span>';
                    break;
                case 'task':
                    title = '<span class="tui-full-calendar-left-content">TASK</span>';
                    break;
                case 'allday':
                    title = '<span class="tui-full-calendar-left-content">ALL DAY</span>';
                    break;
            }

            return title;
        },
        schedule: function(schedule) {

            // use another functions instead of 'schedule'
            // milestone: function() {...}
            // task: function() {...}
            // allday: function() {...}

            var tpl;

            switch(category) {
                case 'milestone':
                    tpl = '<span class="calendar-font-icon ic-milestone-b"></span> <span style="background-color: ' + schedule.bgColor + '">' + schedule.title + '</span>';
                    break;
                case 'task':
                    tpl = '#' + schedule.title;
                    break;
                case 'allday':
                    tpl = getTimeTemplate(schedule, true);
                    break;
            }

            return tpl;
        },
        collapseBtnTitle: function() {
            return '<span class="tui-full-calendar-icon tui-full-calendar-ic-arrow-solid-top"></span>';
        },
        timezoneDisplayLabel: function(timezoneOffset, displayLabel) {
            var gmt, hour, minutes;

            if (!displayLabel) {
                gmt = timezoneOffset < 0 ? '-' : '+';
                hour = Math.abs(parseInt(timezoneOffset / 60, 10));
                minutes = Math.abs(timezoneOffset % 60);
                displayLabel = gmt + getPadStart(hour) + ':' + getPadStart(minutes);
            }

            return displayLabel;
        },
        timegridDisplayPrimayTime: function(time) {
            // will be deprecated. use 'timegridDisplayPrimaryTime'
            var meridiem = 'am';
            var hour = time.hour;

            if (time.hour > 12) {
                meridiem = 'pm';
                hour = time.hour - 12;
            }

            return hour + ' ' + meridiem;
        },
        timegridDisplayPrimaryTime: function(time) {
            var meridiem = 'am';
            var hour = time.hour;

            if (time.hour > 12) {
                meridiem = 'pm';
                hour = time.hour - 12;
            }

            return hour + ' ' + meridiem;
        },
        timegridDisplayTime: function(time) {
            return getPadStart(time.hour) + ':' + getPadStart(time.hour);
        },
        timegridCurrentTime: function(timezone) {
            var templates = [];

            if (timezone.dateDifference) {
                templates.push('[' + timezone.dateDifferenceSign + timezone.dateDifference + ']<br>');
            }

            templates.push(moment(timezone.hourmarker).format('HH:mm a'));

            return templates.join('');
        },
        popupIsAllDay: function() {
            return 'All Day';
        },
        popupStateFree: function() {
            return 'Free';
        },
        popupStateBusy: function() {
            return 'Mission';
        },
        titlePlaceholder: function() {
            return 'Title mission';
        },
        startDatePlaceholder: function() {
            return 'Start date';
        },
        endDatePlaceholder: function() {
            return 'End date';
        },
        popupSave: function() {
            return 'Save';
        },
        popupUpdate: function() {
            return 'Update';
        },
        popupDetailDate: function(isAllDay, start, end) {
            var isSameDate = moment(start).isSame(end);
            var endFormat = (isSameDate ? '' : 'YYYY.MM.DD ') + 'hh:mm a';

            if (isAllDay) {
                return moment(start).format('YYYY.MM.DD') + (isSameDate ? '' : ' - ' + moment(end).format('YYYY.MM.DD'));
            }

            return (moment(start).format('YYYY.MM.DD hh:mm a') + ' - ' + moment(end).format(endFormat));
        },
        popupDetailUser: function(schedule) {
            return 'User : ' + (schedule.attendees || []).join(', ');
        },
        popupDetailState: function(schedule) {
            return 'State : ' + schedule.state || 'Mission';
        },
        popupDetailRepeat: function(schedule) {
            return 'Repeat : ' + schedule.recurrenceRule;
        },
        popupDetailBody: function(schedule) {
            return 'Body : ' + schedule.body;
        },
        popupEdit: function() {
            return 'Edit';
        },
        popupDelete: function() {
            return 'Delete';
        },
        popupDetailLocation: function() {
            return '';
        }
    },
    month: {
        daynames: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
        startDayOfWeek: 1,
        narrowWeekend: true
    },
    week: {
        daynames: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
        startDayOfWeek: 1,
        narrowWeekend: true
    }
});


var lastClickSchedule;

calendar.on('clickSchedule', function(event) {
    var schedule = event.schedule;

    if (lastClickSchedule) {
        calendar.updateSchedule(lastClickSchedule.id, lastClickSchedule.calendarId, {
            isFocused: false
        });
    }
    calendar.updateSchedule(schedule.id, schedule.calendarId, {
        isFocused: true
    });

    lastClickSchedule = schedule;
    events_.forEach(event => {
        //console.log("event: " +  event["id"] + "// scheduler: " + schedule["title"]);
        if(event["id"] == schedule["title"]){
            changeModalState(1, event);
            $("#myModal").modal();
            return;
        }
    });

    // open detail view
});

function onClickTodayBtn() {
    calendar.today();
}

function nextWeek()
{  
    calendar.next();
    chargeMissionsList();
    $("#monthLabel").text(monthNames[calendar.getDate().getMonth()] + " " + calendar.getDate().getFullYear());
    chargeEvents();


}

function previousWeek()
{
    calendar.prev();
    chargeMissionsList();
    $("#monthLabel").text(monthNames[calendar.getDate().getMonth()] + " " + calendar.getDate().getFullYear());
    chargeEvents();

}

function currentWeek(){
    calendar.today();
    chargeMissionsList();
    $("#monthLabel").text(monthNames[calendar.getDate().getMonth()] + " " + calendar.getDate().getFullYear());
    chargeEvents();

}

function newEvent(){
    $("#titleInput").val("");
    $("#timeInput").datepicker("setDate", new Date());

}

function chargeMissionsList(){
    callService('<?php echo $robot_dictionary["service_complex_sequencer_get_sequences"]?>', 'robot_simple_command_manager_msgs/GetCommandManagerList', {}, function (res) {
        command_sequences_selected_ = -1;
        $('#missionSelectInput').empty();
        $('#missionSelectInput').append($('<option>', {
            value: -1,
            text: "--Choose mission--"
            }));
        $('[name=options]').val(-1);
        if(res.success != true){
            alertify.error("Error to charge command sequences. Is it the command sequences running?");
            return;
        }
        else{
            jQuery.each(res["list"]["commands"], function( i, val ) {
                $('#missionSelectInput').append($('<option>', {
                    value: val["id"],
                    text: val["id"]
                }));
            });
        }
    });
}

function chargeEvents()
{
    calendar.clear();
    events_ = null;
    callService('<?php echo $robot_dictionary["service_get_scheduler"]?>', 'robot_simple_command_manager_msgs/GetCommandManagerList', {}, function (res) {
        if(res["list"]["Events"].length!=0){
            events_ = res["list"]["Events"];
            updateEvents();
        }
    });
    
}

function updateEvents()
{
    calendar.clear();
    calendar_structure_ = [];
    
    if(events_.length != 0)
    {
        var day_of_week = -1;
        jQuery.each(events_, function( i, val ) {
            var day_of_week = [];
            var repeat_split = [];
            var repeat_hours = 0;
            var hour = 0;
            var min = 0;
            //List of week
            if(val["wk"].length==0){
                //console.log("wk is empty: I will not add");
                day_of_week = [0,1,2,3,4,5,6];
            }else{
                var i=0;
                for(i; i <= val["wk"].length -1 ; i++){
                    if((val["wk"][i])>=0 && (val["wk"][i])<=6){
                        day_of_week.push(val["wk"][i]);
                    }else
                    {
                        console.log("wk has elements with error: I will use that all days in week" + val["wk"][i]);
                        day_of_week = [0,1,2,3,4,5,6];
                        i = val["wk"].length;
                    }
                }
            }

            //Hour and Minute

            if(!(val["hour"]<24 && val["hour"]>=0)){
                alertify.error("Error to charge " +val["hour"] + " dataTime is wrong. The hours is not between 0 and 23");
                return;
            }else if(!(val["minute"]<60 && val["minute"]>=0)){
                alertify.error("Error to charge " + val["minute"] + " dataTime is wrong. The min is not between 0 and 60");
                return;
            }
            time = val["hour"] + ":" + val["minute"];
            hour = val["hour"];
            min = val["minute"];

            //Normalize date
            

            //Repeat
            repeat_hours = val["repeat_hour"];

            //Endable
            var endable = val["enabled"];

            var mission = val["commands"];

            var date_start = val["date_start"];
            if(date_start == ""){
                date_start = calendar.getDateRangeStart()._date;
            }else{
                date_start = new Date(date_start);
            }
            var date_end = val["date_end"];
            if(date_end == ""){
                date_end = calendar.getDateRangeEnd()._date;
            }else{
                date_end = new Date(date_end);
            }

            var array = getArrayElements(val["id"], date_start, date_end, day_of_week, hour, min, repeat_hours, endable, mission);
            calendar_structure_ = calendar_structure_.concat(array);


        });
        calendar.createSchedules(calendar_structure_);
    }
    else
    {
        alertify.success("Events empty");
    }

}

function normalizeDate(date)
{
    
}


function removeEvent()
{
    jQuery.each(events_, function( i, val ) {
        if(val["id"] == $("#titleInput").val()){
            alertify.confirm('Remove event' + $("#titleInput").val(), 'Do you want remove ' + $("#titleInput").val() + '?', function(){
                srv = {
                    Command: {
                        id: $("#titleInput").val(),
                        command: ''
                    }
                }
                callService('<?php echo $robot_dictionary["service_remove_scheduler"]?>', 'robot_simple_command_manager_msgs/ManageCommandManager', srv, function (res) {
                    if(res["ret"]["success"]){
                        alertify.success("Removed " + $("#titleInput").val() + " event");
                    }else{
                        alertify.error(res["ret"]["message"]);
                    }
                    chargeMissionsList();
                    chargeEvents();
                    $('#myModal').modal('hide');
                });
            }, function(){ 
                alertify.error('Cancel')
            }
            );
        }
    });
}


//start_day: first day to do the event
//end_day: last day to do the event
//day_of_weeks: 0 monday, 1 tuesday, 2 wednesday, 3 thursday, 4 friday, 5 saturday, 6 sunday, 7 all days
//hour: hour to start the event
//min: min to start the event
//repeat: 0 doen't repeat, other the number os hours between the repeat 
function getArrayElements(id, start_day, end_day, days_of_week, hour, min, repeat, endable, mission){
    console.log("Adding: id " + id + " ; start_day: " + start_day + "; end_day: " + end_day + "; days_of_week: " + days_of_week + "; hour: " + hour
        + "; min: " + min + "; repeat: " + repeat + "; mission: " + mission 
    )
    calendar_structure_template = {
        id: '',
        calendarId: '',
        title: '',
        category: 'time',
        dueDateClass: '',
        color: 'blue',
        bgColor: 'red',
        dragBgColor: 'red',
        start: '2021-05-25T01:30:00+09:00',
        end: '2021-05-25T02:30:00+09:00',
        customStyle: ''
    };
    if(!endable){
        calendar_structure_template["bgColor"] = "#d00000";
        calendar_structure_template["dragBgColor"] = "#d00000";
        calendar_structure_template["color"] = "#e00000";
        calendar_structure_template["customStyle"] = "color:#e00000 !important";


        
    }else{
        calendar_structure_template["bgColor"] = "#00d000";
        calendar_structure_template["dragBgColor"] = "#00d000";
        calendar_structure_template["color"] = "#00e000";
        calendar_structure_template["customStyle"] = "color:#00e000 !important";
    }

    if(repeat < 0){
        repeat = abs(repeat);
    }
    array = [];
    start_day = calendarDateToMoment(start_day, hour, min);
    i = 0;
    repeat_times = 0;
    end_day = calendarDateToMoment(end_day, 23, 59);
    iterator_day = Object.assign({}, start_day);
    while(moment(iterator_day).isBefore(moment(end_day))){
        if(days_of_week.includes((((moment(iterator_day).day()+13)%7))))
        {
            calendar_struture_aux = Object.assign({}, calendar_structure_template);
            calendar_struture_aux["id"] = id + "_" + i.toString();
            calendar_struture_aux["calendarId"] = id + "_" + i.toString();
            i++;
            calendar_struture_aux["title"] = id;
            
            if(repeat == 0)
            {
                calendar_struture_aux["start"] = moment(iterator_day).format("YYYY-MM-DD[T]HH:mm:[00]");
                calendar_struture_aux["end"] = moment(iterator_day).format("YYYY-MM-DD[T]HH:mm:[59]");
                array.push(calendar_struture_aux);
                repeat_times = 0;
                iterator_day = moment(iterator_day).add(1,"days").set('hour', hour).set('minute', min);
            }
            else   //repeat != 0 
            { 
                repeat_times ++;
                calendar_struture_aux["start"] = moment(iterator_day).format("YYYY-MM-DD[T]HH:mm:[00]");
                calendar_struture_aux["end"] = moment(iterator_day).format("YYYY-MM-DD[T]HH:mm:[59]");
                array.push(calendar_struture_aux);
                iterator_day_aux = Object.assign({}, iterator_day);
                iterator_day_aux = moment(iterator_day_aux).add(repeat, 'hours');
                if(moment(iterator_day).day()==moment(iterator_day_aux).day()){
                    iterator_day = moment(iterator_day).add(repeat, 'hours');
                    repeat_times ++;
                }
                else{
                    iterator_day = moment(iterator_day).add(1,"days").set('hour', hour).set('minute', min);
                    console.log(iterator_day);
                    repeat_times = 0;
                }
            }

        }
        else
        {
            iterator_day = moment(iterator_day).add(1,"days").set('hour', hour).set('minute', min);
            repeat_times = 0;
        }
    }
    return array;
}

function calendarDateToMoment(date, hour = -1, min = -1){
    var day = ("0" + (date.getDate()).toString()).slice(-2)
    var month = ("0" + (date.getMonth()+1).toString()).slice(-2);
    var year = date.getFullYear();
    var date = year + "-" + month + "-" + day;
    if(hour == -1){
        hour = date.getHours();
    }else{
        hour = ("0" + (hour).toString()).slice(-2);
    }
    if(min == -1){
        min = date.getMinutes();
    }else{
        min = ("0" + (min).toString()).slice(-2);
    }
    var time =  hour + ":" + min + ":00";
    return moment(date).add(hour,'hours').add(min, 'minutes');
}


//calendar.createSchedules([
//    {
//        id: '1',
//        calendarId: '1',
//        title: 'test_add',
//        category: 'time',
//        dueDateClass: '',
//        start: '2021-06-01T01:30:00+09:00',
//        end: '2021-06-01T02:30:00+09:00'
//    },
//    {
//        id: '2',
//        calendarId: '1',
//        title: 'test_add',
//        category: 'time',
//        dueDateClass: '',
//        start: '2021-05-10T08:46:00+09:00',
//        end: '2021-05-10T08:46:59+09:00'
//    }
//]);


</script>
