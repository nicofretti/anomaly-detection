<?php
include "config.php";

include_once 'common.php';

if(isset($_POST['but_submit'])){

    $uname = $_POST['txt_uname'];
    $password = $_POST['txt_pwd'];
//    $uname = mysqli_real_escape_string($con,$_POST['txt_uname']);
//    $password = mysqli_real_escape_string($con,$_POST['txt_pwd']);

    if ($uname != "" && $password != ""){

//        $sql_query = "select count(*) as cntUser from users where usr_username='".$uname."' and usr_password='".$password."'";
//        $result = mysqli_query($con,$sql_query);
//        $row = mysqli_fetch_array($result);

//        $count = $row['cntUser'];

          if ($uname == "admin" & $password == "4dm1N"){
//        if($count == 1){
//            $sql_query = "SELECT * FROM users WHERE usr_username='".$uname."' and usr_password='".$password."'";
//            $result = mysqli_query($con,$sql_query);
//            $row = mysqli_fetch_array($result);
            $_SESSION['uname'] = "admin";
            $_SESSION['id_usr'] = 500;
            $_SESSION['name'] = "admin";
            $_SESSION['admin'] = TRUE;
            $_SESSION['develop'] = TRUE;
            header('Location: index.php');
        }else{
            echo "Invalid username and password";
        }

    }

}
?>

<link rel="stylesheet" href="bootstrap-3.3.7/dist/css/bootstrap.min.css">
<script src="bootstrap-3.3.7/dist/js/bootstrap.min.js"></script>
<!-- jQuery library -->
<script src="jquery/jquery.min.js"></script>
<!-- Latest compiled JavaScript -->

<!DOCTYPE html>
<html>
<head>
	<title><?php echo $lang['PAGE_TITLE']?></title>
        <link rel="icon" href="image/favicon.ico">
   <!--Made with love by Mutiullah Samim -->

	<!--Bootsrap 4 CDN-->
	<link rel="stylesheet" href="bootstrap-3.3.7/dist/css/bootstrap.min.css">

	<!--Custom styles-->
        <link rel="stylesheet" type="text/css" href="css/login.css">
</head>
<body>

<div class="container">
    <img class="img-responsive" src="./image/robotnik_logo.svg" style="margin-top: 100px;" alt="Robotnik">
    <img class="img-responsive" src="./image/ico_robotnik.svg" style="height: 200px; width: 200px; margin: 10% auto" alt="Robotnik">
	<div class="d-flex justify-content-center h-100">
		<div class="card">
			<div class="card-header">
                            <center>
				<h3>Sign In</h3>
                            </center>
			</div>
			<div class="card-body">
				<form method="post" action="">

                                    <div class="input-group">
                                        <span class="input-group-addon"><i class="glyphicon glyphicon-user"></i></span>
                                        <input type="text" class="form-control" id="txt_uname" name="txt_uname" placeholder="Username" />
                                    </div>
                                    <div class="input-group">
                                        <span class="input-group-addon"><i class="glyphicon glyphicon-lock"></i></span>
                                        <input type="password" class="form-control" id="txt_pwd" name="txt_pwd" placeholder="Password"/>
                                    </div>
                                    <div class="form-group">
                                        <center>
                                        <input type="submit" value="Submit" name="but_submit" class="btn login_btn" id="but_submit" />
                                        </center>
                                    </div>
				</form>
			</div>

		</div>
	</div>
</div>
</body>
</html>
