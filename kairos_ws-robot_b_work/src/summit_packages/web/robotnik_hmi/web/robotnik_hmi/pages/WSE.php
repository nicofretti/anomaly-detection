<?php

include './pages/page_dictionary.php';

echo '<body style="background: #f5f5f5">';
include './pages/nav/nav.php'; 

if (sizeof($_GET) == 0) {
    $select_first_page_with_permission = FALSE;
    foreach ($page_dictionary as &$value) {
        if((!$value["only_admin"] and !$select_first_page_with_permission)or ($_SESSION['admin'] and ($value["develop"]!=NULL )and !$select_first_page_with_permission) or ($_SESSION['develop'] and !$select_first_page_with_permission)){
            include $value["path"];
            $select_first_page_with_permission = TRUE;
        }
    }
} else {
    $findpage = FALSE;
    foreach ($page_dictionary as &$value) {
        if ($_GET[$value["key"]] == "true") {
            if((!$value["only_admin"] and ($value["develop"]==NULL or !$value["develop"])) or ($_SESSION['admin'] and ($value["develop"]==NULL or $value["develop"]==FALSE)) or ($_SESSION['develop'])){
                include $value["path"];
                $findpage = TRUE;
            } else {
                include './pages/Error/ErrorNotPermission.php';
                session_destroy();
                header('Location: loging.php');
                $findpage = TRUE;
            }
        }
    }
    if(!$findpage and $_GET["lang"]==NULL){
        include './pages/Error/Error404.php';
    }else if ($_GET["lang"]!=NULL){
        $select_first_page_with_permission = FALSE;
        foreach ($page_dictionary as &$value) {
            if((!$value["only_admin"] and !$select_first_page_with_permission)or ($_SESSION['admin'] and ($value["develop"]!=NULL )and !$select_first_page_with_permission) or ($_SESSION['develop'] and !$select_first_page_with_permission)){
                include $value["path"];
                $select_first_page_with_permission = TRUE;
            }
        }
        
    }
}
if ($_SESSION['admin'] == 1 || $_SESSION['debelop'] == 1) {
            include './pages/nav/user_manager.php'; 
        }
include '</body>';
?>