<?php

$page_dictionary = array();

$page = array("page_name"=>"Monitor", "path"=>"./pages/Monitor/Monitor.php", "key"=>"monitor", "only_admin"=>FALSE);
array_push($page_dictionary, $page);

$page = array("page_name"=>"Sequencer", "path"=>"./pages/CommandSequencer/CommandSequencer.php", "key"=>"CommandSequencer", "only_admin"=>FALSE, "develop"=>False);
array_push($page_dictionary, $page);

$page = array("page_name"=>"Mission", "path"=>"./pages/CommandSequencer/CommandComplex.php", "key"=>"CommandComplex", "only_admin"=>FALSE, "develop"=>False);
array_push($page_dictionary, $page);

$page = array("page_name"=>"Scheduler", "path"=>"./pages/CommandSequencer/CommandScheduler.php", "key"=>"CommandScheduler", "only_admin"=>FALSE, "develop"=>False);
array_push($page_dictionary, $page);

?>
