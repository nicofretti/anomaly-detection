** robotnik_advanced_system_monitor
** requeriments: BBDD (working to have in text)

Description:

Program to get a frame of rosbag when a event it's happens.

Configuration in yaml:

conf_ddbb:
	user - user in DDBB
	password - pasword in DDBB
	host - IP DDBB
	database - Database use in DDBB

config_rasm:
	base_directory - Where it saves the bags
	temporal_bag_directory - When it saves the bag when it is recorded
	saved_bag_directory - When it saves the bag when it has a trigger
	max_size - Max size of saved files, when the folther is more big of this value, it cleans all the files
	force_delet_files_days - When the files is errase automatically
	add_in_data_base - Save the information  in the database
	topics_exclude - List of the topics doesn't recorded in the bag

topics_subscribe: you need say the topics use in your triggers
	alias of the topic, it is neccesary be unique
		topic
		type_msg (package/type)

rasm_logic: The logics of the triggers
	alias of the trigger
		description - A little description of the trigger
		days_of_expiry - How much days it be exist before to delete
		flag_erase_max_size_rasm - It will errase when the max size event exist (use true)
		trigger - Event to record, it need be a logic expression. If you want use a registed topic you need put !--[alias_topic]--! to get the data or get information about the topic ("hz","count","last_msgs" or "start") using /--[alias_topic]--/
			Example:
				!--cmd_vel--!.linear.x > 0 or !--cmd_vel--!.angular.x > 0  #it will trigger if the robot it moving
				/--navpvt--/['hz']<30 and /--navpvt--/['cont']>10 #it will trigger only it has 10 messages and it working under 30 hz
		not_init_in - seconds to wait for start
		not_new_msgs_in - if the topic no fount in n secons
		add_files - list the additional files to save
			alias: path  			
