<!DOCTYPE html>
<!--
To change this license header, choose License Headers in Project Properties.
To change this template file, choose Tools | Templates
and open the template in the editor.
-->
<?php include './common.php' ?>

<html>
    <head>

        <script src="js/eventemitter2.min.js"></script>
        <script src="js/roslib.js"></script>



        <script src="js/three.js"></script>
        <script src="js/ColladaLoader.js"></script>
        <script src="js/STLLoader.js"></script>
        <script src="js/eventemitter2.min.js"></script>

        <script src="js/ros3d.js"></script>

        <!--WARNING! THE IMPORT OF COLLADALOADER MUST BE DONE AFTER THE ROS3D.JS IMPORT otherwise the model is not shown correctly-->
        <script type="text/javascript" src="js/ColladaLoader.js"></script>

        <script type="text/javascript" src="js/virtualjoystick.js"></script>

        <script type="text/javascript" src="js/RosCoreWeb.js"></script>
        <script type="text/javascript" src="js/structMsg.js"></script>
        <script type="text/javascript" src="js/structSrv.js"></script>

        <link rel="stylesheet" href="alertifyjs/css/alertify.min.css"

        <!-- Latest compiled and minified CSS -->
        <link rel="stylesheet" href="bootstrap-3.3.7/dist/css/bootstrap.min.css">

        <!-- jQuery library -->
        <script src="jquery/jquery.min.js"></script>

        <!-- Latest compiled JavaScript -->
        <script src="bootstrap-3.3.7/dist/js/bootstrap.min.js"></script>

        <link rel="stylesheet" href="DataTables-1.10.16/css/jquery.dataTables.min.css">

        <script src="DataTables-1.10.16/js/jquery.dataTables.min.js"></script>


        <script src="alertifyjs/alertify.min.js"></script>

        <link rel="stylesheet" href="css/StyleRobotnik.css">
        <script src="MultiSelect/jquery.selectlistactions.js"></script>

	    <script src="js/vis-js/vis-network.min.js"></script>

        <title><?php echo $lang['PAGE_TITLE']; ?></title>
        <link rel="icon" href="image/favicon.ico">

        <script>

            ros.connect('ws://<?php echo ($_SERVER['SERVER_NAME']); ?>:9090');

            address = 'http://<?php echo  ($_SERVER['SERVER_NAME']); ?>:<?php echo  ($_SERVER['SERVER_PORT']); ?>/robotnik_hmi';

        </script>

        <script src="./js/calendar_component/tui-code-snippet/dist/tui-code-snippet.js"></script>
        <link rel="stylesheet" href="./js/calendar_component/tui-calendar/dist/tui-calendar.css">

        <!-- If you use the default popups, use this. -->
        <link rel="stylesheet" href="./js/calendar_component/tui-date-picker/dist/tui-date-picker.css">
        <link rel="stylesheet" href="./js/calendar_component/tui-time-picker/dist/tui-time-picker.css">
        <script type="text/javascript" src="./js/calendar_component/tui-dom/dist/tui-dom.min.js"></script>
        <script type="text/javascript" src="./js/calendar_component/tui-time-picker/dist/tui-time-picker.min.js"></script>
        <script type="text/javascript" src="./js/calendar_component/tui-date-picker/dist/tui-date-picker.min.js"></script>
        <script type="text/javascript" src="./js/calendar_component/tui-calendar/dist/tui-calendar.min.js"></script>
        <script src="./js/calendar_component/moment.min.js"></script>
        <script src="./js/timepicker/jquery.timepicker.js"></script>
        <link rel="stylesheet" href="./js/timepicker/jquery.timepicker.css">

    </head>

    <?php
    // Check user login or not
    if (!isset($_SESSION['uname'])) {
        header('Location: loging.php');
    }

    // logout
    if (isset($_POST['but_logout'])) {
        session_destroy();
        header('Location: loging.php');
    }
    ?>
    <?php
    include './robot_dictionary.php';
    include './pages/WSE.php';
    ?>
</html>
