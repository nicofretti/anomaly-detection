<nav class="navbar navbar-default navbar-fixed-top">
    <div class="container-fluid">
        <div class="navbar-header">
            <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#navRobotnik">
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
            </button>
            <a class="navbar-brand" href="https://www.robotnik.es/"><img src ="image/logo-robotnik.png" class="img-responsive" style="width: 195px; "/>
            </a>
        </div>
        <div class="collapse navbar-collapse" id="navRobotnik">
            <ul class="nav navbar-nav">
                <?php
                $select_page = FALSE;
                foreach ($page_dictionary as $index => $value) {
                    if((!$value["only_admin"]and ($value["develop"]==NULL or $value["develop"]==FALSE)) or ($_SESSION['admin'] and ($value["develop"]==NULL or $value["develop"]==FALSE)) or ($_SESSION['develop'])){
                        echo "<li id='nav_" . $value["key"] . "' class=";
                        if ($_GET[$value["key"]] == "true" or ( sizeof($_GET) == 0 and !$select_page)) {
                            echo 'active';
                            $select_page = TRUE;
                        } else {
                            echo '';
                        }
                        echo "><a href='?" . $value["key"] . "=true'>" . $value["page_name"] . "</a></li>";
                    }
                }
                ?>

            </ul>
            <ul class="nav navbar-nav navbar-right">
                <li>
                    <a class="dropdown-toggle" data-toggle="dropdown" href="#"><label><?php echo $_SESSION['name'] . " " ?><?php echo $lang['NAV_ACCOUNT'] ?></label><span class="caret"></span></a>
                    <ul class="dropdown-menu">
                        <form method='post' action="">
                            <li>
                                <a onclick="document.getElementById('logout_button').click()">
                                    <span class="glyphicon glyphicon-log-out"></span>
                                    <?php echo $lang['NAV_LOG_OUT']; ?>
                                </a>
                            </li>
                            <li>
                                <a data-toggle="modal" data-target="#modalChangePassword">
                                    <span class="glyphicon glyphicon-lock"></span>
                                    <?php echo $lang['NAV_CHANGE_PASSWORD']; ?>
                                </a>
                            </li>
                            <?php
                            if ($_SESSION['admin'] == 1) {
                                echo '
                                        <li>
                                            <a onclick="openUserManager()">
                                                <span class="glyphicon glyphicon-cog"></span>
                                                ' . $lang['NAV_USER_MANAGER'] . '
                                            </a>
                                        </li>';
                            }
                            ?>
                        </form>
                    </ul>

                </li>
            </ul>
            <ul class="nav navbar-nav navbar-right">
                <li>
                    <a class="dropdown-toggle" data-toggle="dropdown" href="#"><?php echo $lang['NAV_LANGUAGE']; ?><span class="caret"></span></a>
                    <ul class="dropdown-menu">
                        <li><a href="?lang=en"><?php echo $lang['NAV_LANGUAGE_ENGLISH']; ?></a></li>
                        <li><a href="?lang=es"><?php echo $lang['NAV_LANGUAGE_SPANISH']; ?></a></li>
                    </ul>
                </li>
            </ul>

        </div>
    </div>
    <form method='post' action="">
        <button type="submit" value="Logout" name="but_logout" id="logout_button" hidden="true"></button>
    </form>
</nav>

<div class="modal fade" id="modalChangePassword" role="dialog" data-backdrop="static" data-keyboard="false">
    <div class="modal-dialog">
        <!-- Modal content-->
        <div class="modal-content">
            <div class="modal-header">
                <h4 class="modal-title"><?php echo $lang['NAV_CHANGE_PASSWORD']; ?></h4>
            </div>
            <div class="modal-body">
                <label><?php echo $lang['NAV_LAST_PASSWORD']; ?></label>
                <form>
                    <div class="input-group">
                        <span class="input-group-addon"><i class="glyphicon glyphicon-lock"></i></span>
                        <input id="lastPassword" type="password" class="form-control" name="password" placeholder="Password">
                    </div>
                </form>
                <label><?php echo $lang['NAV_NEW_PASSWORD']; ?></label>
                <form>
                    <div class="input-group">
                        <span class="input-group-addon"><i class="glyphicon glyphicon-lock"></i></span>
                        <input id="newPassword1" type="password" class="form-control" name="password" placeholder="Password">
                    </div>
                    <div class="input-group">
                        <span class="input-group-addon"><i class="glyphicon glyphicon-lock"></i></span>
                        <input id="newPassword2" type="password" class="form-control" name="password" placeholder="Password">
                    </div>
                </form>
            </div>
            <div class="modal-footer">
                <div class="row">
                    <div class="col-xs-6 col-sm-6 col-md-6 col-lg-6 text-center">
                        <button type="button" class="btn btn-success" onclick="changePassword()"><?php echo $lang['NAV_CHANGE_PASSWORD_SAVE']; ?></button>
                    </div>
                    <div class="col-xs-6 col-sm-6 col-md-6 col-lg-6 text-center">
                        <button type="button" class="btn btn-danger" data-dismiss="modal"><?php echo $lang['NAV_CHANGE_PASSWORD_CANCEL']; ?></button>
                    </div>
                </div>

            </div>
        </div>
    </div>
</div>

<script>

    function changePassword() {

        if ($('#newPassword1').val() == $('#newPassword2').val() && $('#newPassword1').val() != "") {
            changePasswordDDBB();
        } else if ($('#newPassword1').val() != $('#newPassword2').val()) {
            alertify.error("<?php echo $lang['NAV_MSG_PASSWORD_IS_NOT_THE_SAME']; ?>");
        } else {
            alertify.error("<?php echo $lang['NAV_MSG_PASSWORD_CAN_NOT_EMPTY']; ?>");
        }

    }

    function changePasswordDDBB() {
        var data = {
            "action": "changePassword",
            "request": {
                "id_usr": '<?php echo $_SESSION["id_usr"] ?>',
                "last_password_usr": $('#lastPassword').val(),
                "new_password_usr": $('#newPassword1').val()
            }
        };
        console.log(data);
        data = $(this).serialize() + "&" + $.param(data);
        $.ajax({
            type: "POST",
            dataType: "json",
            url: "bbdd/QueryDDBB.php", //Relative or absolute path to response.php file
            data: data,
            success: function (data) {
            
                missions = data["response"];
                console.log(data);
                if (!data.success) {
                    alertify.error("<?php echo $lang['NAV_MSG_THE_PASSWORD_IS_NOT_CORRECT']; ?>");
                } else {
                    alertify.success("<?php echo $lang['NAV_MSG_THE_PASSWORD_IS_CHANGING']; ?>");
                    $('#modalChangePassword').modal('toggle');
                }
            }
        });
    }

</script>



