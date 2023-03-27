<!-- Modal -->
<div id="modalUserManager" class="modal fade" role="dialog" data-backdrop="static" data-keyboard="false">
  <div class="modal-dialog">

    <!-- Modal content-->
    <div class="modal-content">
      <div class="modal-header">
        
        <h4 class="modal-title"><?php echo $lang['NAV_USER_MANAGER_TITLE'];?></h4>
      </div>
      <div class="modal-body">
        <p><?php echo $lang['NAV_USER_MANAGER_EXPLANATION'];?>
            <button type="button" id="addNewUser" onclick="changeModeManagerUser(3)" class="btn btn-default btn-xs">
            <span class="glyphicon glyphicon-plus"></span><?php echo $lang['NAV_USER_MANAGER_NEW_USER'];?>
            </button>
            <button type="button" id="addDeleteUser" onclick="deleteUserManagerList()" class="btn btn-default btn-xs">
            <span class="glyphicon glyphicon-trash"></span><?php echo $lang['NAV_USER_MANAGER_DELETE_USER'];?>
            </button>
        </p>
        <div class="col-xs-12 col-sm-12 col-md-12 col-lg-12">
            <div class="form-group">
            <label for="#selectUsers"><?php echo $lang['NAV_USER_MANAGER_SELECT_AN_USER']?>:</label>
            <select class="form-control" id="selectUsers">
                <option value=-1><?php echo $lang['NAV_SELECT_A_USER']?></option>
            </select>
        </div>
        <div id="divShowUser" hidden>
            <hr>
            <button type="button" class="btn btn-default btn-sm" id = "buttonModifyUserManager" onclick="changeModeManagerUser(2)" hidden>
                <span class="glyphicon glyphicon-pencil"></span><?php echo $lang['NAV_USER_MANAGER_MODIFY_USER']?>
            </button>
            <div class="form-group">
                <label for="usr"></span><?php echo $lang['NAV_USER_MANAGER_NAME']?>:</label>
                <input type="text" class="form-control" id="inputUsr">
            </div>
            <div class="form-group">
                <label for="usr_name"><?php echo $lang['NAV_USER_MANAGER_USER']?>:</label>
                <input type="text" class="form-control" id="inputUsrName">
            </div>
            <div class="form-group" id="divGroupPassManager">
                <label for="usr_name"><?php echo $lang['NAV_USER_MANAGER_PASSWORD']?>:</label>
                <input type="text" class="form-control" id="inputPassword">
            </div>
            <div class="form-group">
                <label for="cbAdminManager"><?php echo $lang['NAV_USER_MANAGER_ADMIN']?></label>
                <input name="myvalue" id="cbAdmin" type="checkbox"/>
                
            </div>
            
            <hr>
        </div>
      </div>
      <div class="modal-footer">
        <div class="row">
            <div class="col-xs-6 col-sm-6 col-md-6 col-lg-6 text-center">
                <button id="buttonSaveUser" type="button" class="btn btn-success" onclick="saveUserManager()"><?php echo $lang['NAV_CHANGE_PASSWORD_SAVE']; ?></button>
            </div>
            <div class="col-xs-6 col-sm-6 col-md-6 col-lg-6 text-center">
                <button type="button" class="btn btn-danger" id="buttonCancel" onclick="openUserManager()" ><?php echo $lang['NAV_CHANGE_PASSWORD_CANCEL']; ?></button>
                <button type="button" class="btn btn-primary" id="buttonClose" data-dismiss="modal"><?php echo $lang['NAV_CHANGE_PASSWORD_CLOSE']; ?></button>
            </div>
        </div>
      </div>
    </div>

  </div>
</div>

<script>


var modeManager=-1;

function changeModeManagerUser(mode){
    if(mode == 0){
        $('#buttonCancel').hide();
        $('#buttonSaveUser').hide();
        $('#divShowUser').hide();
        $('#buttonModifyUserManager').hide();
        $('#addNewUser').show();
        $('#buttonClose').show();
        $('#addDeleteUser').hide();
        $('#divGroupPassManager').hide();
    }else if (mode == 1){
        
        $('#addNewUser').show();
        $('#inputUsrName').val($("#selectUsers option:selected").attr("data-username_usr"));
        $('#inputUsrName').prop('disabled', true);
        $('#inputUsr').val($("#selectUsers option:selected").attr("data-name_usr"));
        $('#inputUsr').prop('disabled', true);
        $('#inputPassword').val($("#selectUsers option:selected").attr("data-password"));
        $('#inputPassword').prop('disabled', true);
        $('#cbAdmin').attr('disabled',true);
        if ($("#selectUsers option:selected").attr("data-admin")==1){
            document.getElementById("cbAdmin").checked = true;
        }else{
            document.getElementById("cbAdmin").checked = false;
        }
        $('#divShowUser').show();
        $('#buttonModifyUserManager').show();
        $('#buttonCancel').hide();
        $('#buttonSaveUser').hide();
        $('#buttonClose').show();
        $('#addDeleteUser').hide();
        $('#divGroupPassManager').hide();
        
    }else if (mode == 2){
        $('#addNewUser').hide();
        $('#inputUsrName').prop('disabled', false);
        $('#inputUsr').prop('disabled', false);
        $('#inputPassword').prop('disabled', false);
        $('#cbAdmin').attr('disabled',false);
        $('#buttonModifyUserManager').show();
        $('#buttonClose').hide();
        $('#buttonCancel').show();
        $('#buttonSaveUser').show();
        $('#buttonModifyUserManager').hide();
        $('#addDeleteUser').show();
        $('#divGroupPassManager').show();
        
    }else if (mode == 3){
        $('#addNewUser').hide();
        $('#inputUsrName').prop('disabled', false);
        $('#inputUsrName').val("");
        $('#inputUsr').prop('disabled', false);
        $('#inputUsr').val("")
        $('#inputPassword').prop('disabled', false);
        $('#inputPassword').val("")
        $('#cbAdmin').attr('disabled',false);
        $('#cbAdmin').attr('checked',false);
        $('#buttonModifyUserManager').hide();
        $('#buttonClose').hide();
        $('#buttonCancel').show();
        $('#buttonSaveUser').show();
        $('#divShowUser').show();
        $('#divGroupPassManager').show();
    }
    modeManager = mode;
}




function saveUserManager(){
    if(modeManager==2){
        updateUserManager();
    }else if(modeManager==3){
        newUserManager();
    }else{
        return;
    }
}


function openUserManager() {
    updateUserManagerList();
    $('#modalUserManager').modal('show');
    changeModeManagerUser(0);
}

function updateUserManagerList(){
    var data = {
        "action": "getUsers",
        "request": {
            "id_admin_usr": '<?php echo $_SESSION["id_usr"] ?>'
        }
    };
    data = $(this).serialize() + "&" + $.param(data);
    $.ajax({
        type: "POST",
        dataType: "json",
        url: "bbdd/QueryDDBB.php", //Relative or absolute path to response.php file
        data: data,
        success: function (data) {
            missions = data["response"];
            if (!data.success) {
                alertify.error("<?php echo $lang['TXT_ERROR_BBDD']; ?>" + data["msg"]);
            } else {
                $('#selectUsers').empty();
                $('#selectUsers').append($('<option>', {
                    value: -1,
                    text: "<?php echo $lang['NAV_CHOOSE_A_USER']; ?>"
                }));
                $('[name=options]').val(-1);
                select_user = -1;
                jQuery.each(data["response"], function (i, val) {
                    $('#selectUsers').append("<option value= '" + val["id_usr"] +
                        "' data-id_usr = " + val["id_usr"] +
                        " data-password = ''" +
                        " data-username_usr = '" + val["username_usr"] +
                        "' data-name_usr = '" + val["name_usr"] +
                        "' data-admin = '" + val["admin_usr"] +
                        "' >" + val["name_usr"] + "</option>");
                });
                
            }
        }
    });

}


function updateUserManager(){
    if($('#inputUsrName').val()==""){
        alertify.error("<?php echo $lang['NAV_USER_MANAGER_ADD_ERROR_USER_NAME']; ?>");
        return;
    }else if($('#inputUsr').val()==""){
        alertify.error("<?php echo $lang['NAV_USER_MANAGER_ADD_ERROR_USER']; ?>");
        return;
    }else if($('#inputPassword').val().length<5 && $('#inputPassword').val()!=""){
        alertify.error("<?php echo $lang['NAV_USER_MANAGER_ADD_ERROR_LENGTH_PASSWORD']; ?>");
        return;
    }
    var data = {
        "action": "updateUser",
        "request": {
            "id_admin_usr": <?php echo $_SESSION["id_usr"] ?>,
            "id_usr": $("#selectUsers option:selected").attr("data-id_usr"),
            "user_usr": $('#inputUsr').val(),
            "name_usr": $('#inputUsrName').val(),
            "password_usr": $('#inputPassword').val(),
            "admin_usr": (document.getElementById("cbAdmin").checked?1:0)
        }
    };
    data = $(this).serialize() + "&" + $.param(data);
    $.ajax({
        type: "POST",
        dataType: "json",
        url: "bbdd/QueryDDBB.php", //Relative or absolute path to response.php file
        data: data,
        success: function (data) {
            if (!data.success) {
                alertify.error("<?php echo $lang['TXT_ERROR_BBDD']; ?>" + data["msg"]);
            } else {
                alertify.success("<?php echo $lang['NAV_USER_MANAGER_UPDATE_USER']; ?>");
                openUserManager();
            }
        }
    });

}


function newUserManager(){
    if($('#inputUsrName').val()==""){
        alertify.error("<?php echo $lang['NAV_USER_MANAGER_ADD_ERROR_USER_NAME']; ?>");
        return;
    }else if($('#inputUsr').val()==""){
        alertify.error("<?php echo $lang['NAV_USER_MANAGER_ADD_ERROR_USER']; ?>");
        return;
    }else if($('#inputPassword').val()==""){
        alertify.error("<?php echo $lang['NAV_USER_MANAGER_ADD_ERROR_PASSWORD']; ?>");
        return;
    }else if($('#inputPassword').val().length<5){
        alertify.error("<?php echo $lang['NAV_USER_MANAGER_ADD_ERROR_LENGTH_PASSWORD']; ?>");
        return;
    }
    var data = {
        "action": "addUser",
        "request": {
            "id_admin_usr": <?php echo $_SESSION["id_usr"] ?>,
            "user_usr": $('#inputUsr').val(),
            "name_usr": $('#inputUsrName').val(),
            "password_usr": $('#inputPassword').val(),
            "admin_usr": (document.getElementById("cbAdmin").checked?1:0)
        }
    };
    data = $(this).serialize() + "&" + $.param(data);
    $.ajax({
        type: "POST",
        dataType: "json",
        url: "bbdd/QueryDDBB.php", //Relative or absolute path to response.php file
        data: data,
        success: function (data) {
            if (!data.success) {
                alertify.error("<?php echo $lang['TXT_ERROR_BBDD']; ?>" + data["msg"]);
            } else {
                alertify.success("<?php echo $lang['NAV_USER_MANAGER_ADD_USER']; ?>");
                openUserManager();
            }
        }
    });
}


function deleteUserManagerList(){

  alertify.confirm("<?php echo $lang['NAV_USER_MANAGER_ERRASE_DIALOG']; ?>",
  function(){
    var data = {
        "action": "deleteUser",
        "request": {
            "id_admin_usr": <?php echo $_SESSION["id_usr"] ?>,
            "id_usr": $("#selectUsers option:selected").attr("data-id_usr")
        }
    };
    data = $(this).serialize() + "&" + $.param(data);
    $.ajax({
        type: "POST",
        dataType: "json",
        url: "bbdd/QueryDDBB.php", //Relative or absolute path to response.php file
        data: data,
        success: function (data) {
            missions = data["response"];
            if (!data.success) {
                alertify.error("<?php echo $lang['TXT_ERROR_BBDD']; ?>" + data["msg"]);
            } else {
                alertify.success("<?php echo $lang['NAV_USER_MANAGER_ERRASE_USER']; ?>");
                openUserManager();
            }
        }
    });
  },
  function(){
    return;
  }).setHeader('<?php echo $lang['NAV_USER_MANAGER_ERRASE_USER']; ?>');
    

}


var userSelected = undefined;

$("#selectUsers").change(function () {
    $( "#selectUsers option:selected" ).each(function() {
        user_id = $( this ).val();
    });
    userSelected = $("#listUsers option:selected");
    if(user_id == -1){
        changeModeManagerUser(0);
    }else{
        changeModeManagerUser(1);
    }
}).change();



</script>

