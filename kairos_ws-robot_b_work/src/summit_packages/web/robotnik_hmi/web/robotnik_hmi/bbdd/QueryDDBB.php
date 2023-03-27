<?php


$servername = "127.0.0.1";
//$servername = "192.168.10.97";
$username = "robotnik";
$password = "R0b0tn1K";
$dbname = "db_robotnik";

// Create connection
try {
    $conn = new mysqli($servername, $username, $password, $dbname);
}catch (Exception $e){
    error_function("Exception: " . $e->getMessage());
}

if (is_ajax()) {
    if ($conn->connect_error){
        error_function("Connection failed: " . $conn->connect_error);
    }
    else if (isset($_POST["action"]) && !empty($_POST["action"])) { //Checks if action value exists
        $action = $_POST["action"];
        try {
            switch($action) { //Switch case for value of action
                case "test":
                case "info":
                case "version":
                    test_function();
                    break;
                case "changePassword":
                    if (has_request()){
                        changePassword($_POST["request"]);
                    }
                    break;
                case "getUsers":
                    if (has_request()){
                        getUsers($_POST["request"]);
                    }
                    break;
                case "updateUser":
                    if (has_request()){
                        updateUser($_POST["request"]);
                    }
                    break;
                case "deleteUser":
                    if (has_request()){
                        deleteUser($_POST["request"]);
                    }  
                    break;
                case "addUser":
                    if (has_request()){
                        addUser($_POST["request"]);
                    }  
                    break;
                default :
                    error_function("Action not defined - " . $action);
                    break;
            }
        } catch (Exception $e){
            error_function("Exception: " . $e->getMessage());
        }
    }else{
        error_function("Action empty");
    }
}

function cleanStringSpecialChars($string) {
   $string = preg_replace('/[^A-Za-z0-9().\n\-_ :><?¿¡!]/', '', $string); // Removes special chars.
   return preg_replace('/-+/', '-', $string); // Replaces multiple hyphens with single one.
}

function hasAnySpecialChar($string)
{
    $normalizedString = cleanStringSpecialChars($string);
    return $normalizedString != $normalizedString;
}

//Function to check if the request is an AJAX request
function is_ajax() {
    return isset($_SERVER['HTTP_X_REQUESTED_WITH']) && strtolower($_SERVER['HTTP_X_REQUESTED_WITH']) == 'xmlhttprequest';
}

function has_request(){
    if(isset($_POST["request"]) && !empty($_POST["request"])){
        return true;
    }else{
        error_function("Request empty");
        return false;
    }
}

function has_value($val, $tag){
    if(isset($val[$tag]) && (($val[$tag])!=null)){
        return true;
    }else{
        return false;
    }
}

function error_function($msg){
    $return["success"] = false;
    $return["msg"] = $msg;
    $return["response"] = NULL;
    echo json_encode($return);
}

function error_function_obj($msg, $obj){
    $return["success"] = false;
    $return["msg"] = $msg;
    $return["response"] = $obj;
    echo json_encode($return);
}

function success_function($data){
    $return["success"] = true;
    $return["msg"] = "Success";
    $return["response"] = $data;
    echo json_encode($return);
}

function test_function(){
    $ret["developer"] = "Juanma";
    $ret["company"] = "Robotnik";
    $ret["version"] = "v1.0";
    success_function($ret);
}


function changePassword($request){
    if (!has_value($request, "id_usr") || (!is_numeric($request["id_usr"])||!is_int((int)$request["id_usr"]))){
        error_function_obj("Request not found - id_usr" , !is_int((int)$request["id_usr"]));
        return;
    }
    if (!has_value($request, "last_password_usr")){
        error_function_obj("Request not found - last_password_usr" , $request["last_password_usr"]);
    }
    $request["last_password_usr"] = cleanStringSpecialChars($request["last_password_usr"]);
    if (!has_value($request, "new_password_usr")){
        error_function_obj("Request not found - new_password_usr" , $request["new_password_usr"]);
        return;
    } else if($request["new_password_usr"] == ""){
        error_function_obj("The password can't be empty" , $request["new_password_usr"]);
        return;
    }else if(hasAnySpecialChar($request["new_password_urs"])){
        error_function_obj("The password only accept chars, numbers, (, ), ., \-, :, >, <" , $request["new_password_usr"]);
        return;
    }
    global $conn;

    $sql = "SELECT * FROM `users` "
            ."WHERE usr_id = '" . $request["id_usr"] . "' AND usr_password = '" . $request["last_password_usr"] . "'";
    $resultQuerry = $conn->query($sql);
    $result = NULL;
    if ($resultQuerry->num_rows > 0) {
        $sql = "UPDATE `users` SET "
            . "`usr_password` = '" . $request["new_password_usr"] ."' "
            . "WHERE `users`.`usr_id` = '" . $request["id_usr"] . "'";
        $resultUpdate = $conn->query($sql);
        if ($resultUpdate === FALSE) {
            error_function("Not update");
        }else{
            success_function("Update");
        }
    }else{
        error_function("Request not found - Not update, the user and password not exist");
    }
}

function updateUser($request){
    $psw_sql = "";
    if (!has_value($request, "id_admin_usr") || (!is_numeric($request["id_admin_usr"])||!is_int((int)$request["id_admin_usr"]))){
        error_function_obj("Request not found - id_admin_usr" , !is_int((int)$request["id_admin_usr"]));
        return;
    }
    if (!has_value($request, "user_usr") && $request["user_usr"]!=""){
        error_function_obj("Request not found - user_usr" , $request["user_usr"]);
        return;
    }
    if (!has_value($request, "id_usr") || (!is_numeric($request["id_usr"])||!is_int((int)$request["id_usr"]))){
        error_function_obj("Request not found - id_usr" , !is_int((int)$request["id_usr"]));
        return;
    }
    if (!has_value($request, "user_usr") && $request["user_usr"]!=""){
        error_function_obj("Request not found - user_usr" , $request["user_usr"]);
        return;
    }
    $request["user_usr"] = cleanStringSpecialChars($request["user_usr"]);
    if (!has_value($request, "name_usr") && $request["name_usr"]!=""){
        error_function_obj("Request not found - name_usr" , $request["user_usr"]);
        return;
    }
    $request["name_usr"] = cleanStringSpecialChars($request["name_usr"]);
    if(has_value($request, "password_usr") && hasAnySpecialChar($request["password_usr"])){
        error_function_obj("The password only accept chars, numbers, (, ), ., \-, :, >, <" , $request["password_usr"]);
        return;
    }else if (has_value($request, "password_usr")){
        if (strlen($request["password_usr"])>5 and $request["password_usr"]!=""){
            $request["password_usr"] = cleanStringSpecialChars($request["password_usr"]);
            $psw_sql = "`usr_password`='".$request["password_usr"]. "', ";
        }else {
            error_function_obj("The password needs to have more 5 chars" , $request["password_usr"]);
            return;
        }
    }
    
    if (!has_value($request, "admin_usr") || (!is_numeric($request["admin_usr"])||!is_int((int)$request["admin_usr"]))){
        error_function_obj("Request not found - admin_usr" , !is_int((int)$request["admin_usr"]));
        return;
    }
    
    global $conn;

    $sql = "SELECT * FROM `users` "
            ."WHERE usr_id = '" . $request["id_admin_usr"]. "'";
    $resultQuerry = $conn->query($sql);
    $result = NULL;

    if ($resultQuerry->num_rows > 0) {
        $sql = "SELECT * FROM `users` "
            ."WHERE usr_name = '" . $request["usr_name"]. "'";
        $resultQuerry = $conn->query($sql);
        if($resultQuerry->num_rows > 0){
            error_function("The user exist");
            return;
        }
        $sql = "UPDATE users SET "
            . "`usr_name` ='". $request["user_usr"] . "', "
            . "`usr_username` ='". $request["name_usr"] . "', "
            . $psw_sql
            . "`usr_admin` = '". $request["admin_usr"] . "' "
            . "WHERE `usr_id` = '". $request["id_usr"] . "'";
        $resultUpdate = $conn->query($sql);
        if ($resultUpdate === FALSE) {
            error_function("Not added");
        }else{
            success_function("Added");
        }
    }else{
        error_function("Request not found - Not update, the admin user not exist");
    }
}

function getUsers($request){
    if (!has_value($request, "id_admin_usr") || (!is_numeric($request["id_admin_usr"])||!is_int((int)$request["id_admin_usr"]))){
        error_function_obj("Request not found" , !is_int((int)$request["id_admin_usr"]));
        return;
    }
    global $conn;
    $sql = "SELECT * FROM `users` "
            ."WHERE usr_id = '" . $request["id_admin_usr"]. "'";
    $resultQuerry = $conn->query($sql);
    $result = array();
    $aux = NULL;
    if ($resultQuerry->num_rows > 0) {
        $sql = "SELECT * FROM `users` WHERE usr_id <> '" . $request["id_admin_usr"]. "' AND usr_develop = '0'";
        $resultQuerry = $conn->query($sql);
        if ($resultQuerry->num_rows > 0){
            while($row = $resultQuerry->fetch_assoc()){
                $aux["id_usr"] = (int)$row["usr_id"];
                $aux["username_usr"] = $row["usr_username"];
                $aux["name_usr"] = $row["usr_name"];
                $aux["admin_usr"] = (int)$row["usr_admin"];
                array_push($result, $aux);
            }

        }
        success_function($result);
    }else{
        error_function("Request not found - Not update, the admin user not exist" );
    }

}

function addUser($request){
    if (!has_value($request, "id_admin_usr") || (!is_numeric($request["id_admin_usr"])||!is_int((int)$request["id_admin_usr"]))){
        error_function_obj("Request not found - id_admin_usr" , !is_int((int)$request["id_admin_usr"]));
        return;
    }
    if (!has_value($request, "user_usr") && $request["user_usr"]!=""){
        error_function_obj("Request not found - user_usr" , $request["user_usr"]);
        return;
    }
    $request["user_usr"] = cleanStringSpecialChars($request["user_usr"]);
    if (!has_value($request, "name_usr") && $request["name_usr"]!=""){
        error_function_obj("Request not found - name_usr" , $request["user_usr"]);
        return;
    }
    $request["name_usr"] = cleanStringSpecialChars($request["name_usr"]);
    if (!has_value($request, "password_usr")){
        error_function_obj("Request not found - password_usr" , $request["password_usr"]);
        return;
    } else if($request["password_usr"] == ""){
        error_function_obj("The password couldn't be empty" , $request["password_usr"]);
        return;
    }else if(hasAnySpecialChar($request["password_usr"])){
        error_function_obj("The password only accept chars, numbers, (, ), ., \-, :, >, <" , $request["password_usr"]);
        return;
    }else if (strlen($request["password_usr"])<5){
        error_function_obj("The password needs to have more 5 chars" , $request["password_usr"]);
        return;
    }
    $request["password_usr"] = cleanStringSpecialChars($request["password_usr"]);
    if (!has_value($request, "admin_usr") || (!is_numeric($request["admin_usr"])||!is_int((int)$request["admin_usr"]))){
        error_function_obj("Request not found - admin_usr" , !is_int((int)$request["admin_usr"]));
        return;
    }
    
    global $conn;

    $sql = "SELECT * FROM `users` "
            ."WHERE usr_id = '" . $request["id_admin_usr"]. "'";
    $resultQuerry = $conn->query($sql);
    $result = NULL;

    if ($resultQuerry->num_rows > 0) {
        $sql = "SELECT * FROM `users` "
            ."WHERE usr_username = '" . $request["name_usr"]. "'";
        $resultQuerry = $conn->query($sql);
        if($resultQuerry->num_rows > 0){
            error_function("The user exist");
            return;
        }
        $sql = "INSERT INTO users (usr_username, usr_name, usr_password, usr_admin) VALUES ("
            . "'". $request["name_usr"] . "', "
            . "'". $request["user_usr"] . "', "
            . "'". $request["password_usr"] . "', "
            . "'". $request["admin_usr"] . "')";
        $resultUpdate = $conn->query($sql);
        if ($resultUpdate === FALSE) {
            error_function("Not added");
        }else{
            success_function("Added");
        }
    }else{
        error_function("Request not found - Not update, the admin user not exist");
    }
}

function deleteUser($request){
    if (!has_value($request, "id_admin_usr") || (!is_numeric($request["id_admin_usr"])||!is_int((int)$request["id_admin_usr"]))){
        error_function_obj("Request not found - id_admin_usr" , !is_int((int)$request["id_admin_usr"]));
        return;
    }
    if (!has_value($request, "id_usr") || (!is_numeric($request["id_usr"])||!is_int((int)$request["id_usr"]))){
        error_function_obj("Request not found - id_usr" , !is_int((int)$request["id_usr"]));
        return;
    }
    if ($request["id_usr"] == $request["id_admin_usr"]){
        error_function_obj("The user can't be eliminated by the same user");
        return;
    }

    global $conn;

    $sql = "SELECT * FROM `users` "
            ."WHERE `usr_id` = '" . $request["id_admin_usr"] . "'";
    $resultQuerry = $conn->query($sql);
    $result = NULL;

    if ($resultQuerry->num_rows > 0) {
        $sql = "DELETE FROM `users` WHERE `usr_id` = '" . $request["id_usr"] . "'";
        $resultQuerry = $conn->query($sql);
        success_function("Errase");

    }else{
        error_function("Request not found - Not update, the admin user not exist");
    }
}


?>
