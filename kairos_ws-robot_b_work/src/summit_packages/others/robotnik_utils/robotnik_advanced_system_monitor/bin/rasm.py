#!/usr/bin/env python
# license removed for brevity

import rospy

#To call proccess
import os
from subprocess import Popen

import copy
from os import remove
import re


#Function to insert data in DDBB:
from datetime import date, datetime, timedelta
import mysql.connector

from shutil import copyfile


#register a new Rasm alarm in ddbb
def insert_rasm_in_ddbb (name_rasm , description_rasm, directory_log_rasm, init_bag_rasm, end_bag_rasm, insert_rasm, date_of_expiry_rasm=(datetime.now() + timedelta(days=1)), flag_erase_date_expiry_rasm=True, flag_erase_max_size_rasm=True):
    
    if(flag_erase_date_expiry_rasm == True):
        flag_erase_date_expiry_rasm = 1
    else:
        flag_erase_date_expiry_rasm = 0
    if(flag_erase_max_size_rasm == True):
        flag_erase_max_size_rasm = 1
    else:
        flag_erase_max_size_rasm = 0
    global conf_ddbb
    cnx = mysql.connector.connect(user=conf_ddbb["user"], password=conf_ddbb["password"], host=conf_ddbb["host"], database=conf_ddbb["database"])
    cursor = cnx.cursor()
    add_rasm = ("INSERT INTO rasm "
        "(name_rasm, description_rasm, init_bag_rasm, end_bag_rasm, directory_log_rasm, insert_rasm, date_of_expiry_rasm, flag_erase_date_expiry_rasm, flag_erase_max_size_rasm) "
        "VALUES (%(name_rasm)s, %(description_rasm)s, %(init_bag_rasm)s, %(end_bag_rasm)s, %(directory_log_rasm)s, %(insert_rasm)s, %(date_of_expiry_rasm)s, %(flag_erase_date_expiry_rasm)s, %(flag_erase_max_size_rasm)s)")
    # Insert rasm
    
    data_rasm = {
        'name_rasm': name_rasm,
        'description_rasm': description_rasm,
        'init_bag_rasm': init_bag_rasm.strftime("%Y-%m-%d %H:%M"),
        'end_bag_rasm': end_bag_rasm.strftime("%Y-%m-%d %H:%M"),
        'directory_log_rasm': directory_log_rasm,
        'insert_rasm': insert_rasm,
        'date_of_expiry_rasm': date_of_expiry_rasm.strftime("%Y-%m-%d %H:%M"),
        'flag_erase_date_expiry_rasm': flag_erase_date_expiry_rasm,
        'flag_erase_max_size_rasm': flag_erase_max_size_rasm
    }
    cursor.execute(add_rasm, data_rasm)
    
    # Make sure data is committed to the database
    cnx.commit()
    cursor.close()
    cnx.close()
    
#Get all rasm expired
def ddbb_get_expired():
    global conf_ddbb
    cnx = mysql.connector.connect(user=conf_ddbb["user"], password=conf_ddbb["password"], host=conf_ddbb["host"], database=conf_ddbb["database"])
    cursor = cnx.cursor()

    query = (" SELECT directory_log_rasm FROM `rasm` WHERE date_of_expiry_rasm < Now() AND (`erase_for_date_expiry_rasm`<>1 OR `erase_for_max_size_rasm` <> 1) AND `flag_erase_date_expiry_rasm` = 1")
    cursor.execute(query)
    
    myresult = cursor.fetchall()

    register = []
    for x in myresult:
        register.append(x[0])
    
    cursor.close()
    cnx.close()
    return register

def ddbb_get_expired_max_time():
    global conf_ddbb
    global config_rasm
    cnx = mysql.connector.connect(user=conf_ddbb["user"], password=conf_ddbb["password"], host=conf_ddbb["host"], database=conf_ddbb["database"])
    cursor = cnx.cursor()

    query = ("SELECT directory_log_rasm FROM `rasm` WHERE (date_of_expiry_rasm < NOW() - INTERVAL "+str(config_rasm["force_delet_files_days"])+" DAY) AND (`erase_for_date_expiry_rasm`<>1 OR `erase_for_max_size_rasm` <> 1)")
    cursor.execute(query)
    myresult = cursor.fetchall()
    register = []
    for x in myresult:
        register.append(x[0])
    

    cursor.close()
    cnx.close()
    return register

def ddbb_update_expired (directory_log_rasm):
    
    cnx = mysql.connector.connect(user=conf_ddbb["user"], password=conf_ddbb["password"], host=conf_ddbb["host"], database=conf_ddbb["database"])
    cursor = cnx.cursor()
    query = ("UPDATE rasm SET erase_for_max_size_rasm=1 WHERE directory_log_rasm=(directory_log_rasm)%s")
    # Insert rasm
    
    data_query = {
        'directory_log_rasm': directory_log_rasm,
    }
    cursor.execute(query, data_query)
    
    # Make sure data is committed to the database
    cnx.commit()
    cursor.close()
    cnx.close()


#Get all rasm with flag max size


def ddbb_get_max_size():
    global conf_ddbb
    cnx = mysql.connector.connect(user=conf_ddbb["user"], password=conf_ddbb["password"], host=conf_ddbb["host"], database=conf_ddbb["database"])
    cursor = cnx.cursor()

    query = ("SELECT directory_log_rasm FROM `rasm` WHERE (`erase_for_date_expiry_rasm`<>1 OR `erase_for_max_size_rasm` <> 1) AND `flag_erase_max_size_rasm`=1")
    cursor.execute(query)
    myresult = cursor.fetchall()
    register = []
    for x in myresult:
        register.append(x[0])
    
    cursor.close()
    cnx.close()
    return register

def ddbb_update_rasm_max_size (directory_log_rasm):
    
    cnx = mysql.connector.connect(user=conf_ddbb["user"], password=conf_ddbb["password"], host=conf_ddbb["host"], database=conf_ddbb["database"])
    cursor = cnx.cursor()
    query = ("UPDATE rasm SET erase_for_max_size_rasm=1 WHERE directory_log_rasm=(directory_log_rasm)%s")
    # Insert rasm
    
    data_query = {
        'directory_log_rasm': directory_log_rasm,
    }
    cursor.execute(query, data_query)
    
    # Make sure data is committed to the database
    cnx.commit()
    cursor.close()
    cnx.close()

def ddbb_update_rasm_delete_all():
    cnx = mysql.connector.connect(user=conf_ddbb["user"], password=conf_ddbb["password"], host=conf_ddbb["host"], database=conf_ddbb["database"])
    cursor = cnx.cursor()
    query = ("UPDATE rasm SET erase_for_max_size_rasm=1")
    # Insert rasm
    
    cursor.execute(query)
    
    # Make sure data is committed to the database
    cnx.commit()
    cursor.close()
    cnx.close()

#Auto generate CallBacks

currentCallBacks = {}
informationCallBacks = {}

def createCallBack (nameCallBack, aliasData):
    global currentCallBacks
    global informationCallBacks
    currentCallBacks [aliasData] = None
    informationCallBacks [aliasData] = {
        'start' : datetime.now(),
        'last_msg': None,
        'hz': None,
        'count': 0
    }
    defCallBack = """def callBack"""+nameCallBack+""" (data):
    
    if informationCallBacks ['"""+aliasData+"""']["count"] > 0:
        informationCallBacks ['"""+aliasData+"""']["hz"] = 1/(datetime.now() - informationCallBacks ['"""+aliasData+"""']["last_msg"]).total_seconds()
    informationCallBacks ['"""+aliasData+"""']["last_msg"] = datetime.now()
    informationCallBacks ['"""+aliasData+"""']["count"]  += 1
    currentCallBacks['"""+aliasData+"""'] = data
    """
    return defCallBack


def generalSubscribe (topic, callback, type_msg, aliasData):
    msg_pkg = type_msg.split("/")[0]+".msg"
    msg_type = type_msg.split("/")[1]
    import_msg_sentence = """from """+msg_pkg+""" import """+ msg_type
    exec (import_msg_sentence)
    callback_sentence = createCallBack(callback, aliasData)
    exec (callback_sentence)
    subscribe_sentence = """rospy.Subscriber ('"""+ topic +"""', """ + msg_type + """, callBack""" + callback + """)"""
    exec (subscribe_sentence)


#Timer logic triggers

def callBackTriggerLogic(event):
    checkRules()


def checkRules():
    global rules_rasm
    global currentCallBacks;
    global informationCallBacks;
    global bagDescription
    if (bagDescription["path"] != None):
        for rule in rules_rasm:
            alltopicsactive = True
            for key, value in rule["topics_used"].iteritems():
                if currentCallBacks[key] == None:
                    alltopicsactive = False
            if (alltopicsactive == True and rule["trigger"]!=""):
                if eval(rule["trigger"]) == True and rule ["active"] == False:
                    rule ["init_time"] = datetime.now()
                    rule ["active"] = True
                    rule ["saved"] = False
                #print (rule["name"] + "-" + str(rule["active"]))
            #Si se quiere anhadir lo de no iniciado queda aqui


#Check alarms and save
def callBackTriggerBag(event):
    global bagDescription
    if (bagDescription["init_bag"] != None):
        bagDescription["end_bag"] =  datetime.now()
        registAlarm()
        try:
            remove(bagDescription["path"])
        except OSError as e:
            print (e)
    bagDescription["path"] = config_rasm["temporal_bag_directory"] +"/"+datetime.now().strftime("%Y-%m-%d_%H-%M-%s")+".bag"
    print ("Init bag at: " + datetime.now().strftime("%Y-%m-%d_%H.%M.%s"))
    exclude_topics = ""
    for index, topic_e in enumerate(config_rasm["topics_exclude"]):
        exclude_topics = exclude_topics + topic_e
        if(index+1 != len(config_rasm["topics_exclude"])):
            exclude_topics = exclude_topics + "|"
    print('--exclude="' + exclude_topics + '"')
    p = Popen(["rosbag", "record", "-a", "-x", exclude_topics, "--output-name="+bagDescription["path"], "--duration="+str(config_rasm["frame_bag"])])
    bagDescription["init_bag"] = datetime.now()
    

            
def registAlarm():
    global rules_rasm
    for rule in rules_rasm:
        if rule["active"]==True and rule ["saved"] == False:
            saveAlarm(rule["name"], rule["description"], rule["init_time"], rule["days_of_expiry"], rule["flag_expired_time"], rule["flag_erase_max_size_rasm"], rule["add_files"])
            rule["saved"] = True
            rule ["init_time"] = None
            rule["active"]= False

def saveAlarm (name, description, init_time, days_of_expiry, flag_expired, flag_max_size, additional_files):
    global config_rasm
    global rules_rasm
    directory = config_rasm["saved_bag_driectory"]+"/"+name+"/"+init_time.strftime("%Y-%m-%d_%H.%M.%s")
    try:
        print("Saved alarm " + name + " in " + directory)
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            print ("Created -> " + directory)
        else:
            print ("The directory exist -> " + directory)
        dest = directory +"/"+datetime.now().strftime("%Y-%m-%d_%H-%M")+".bag"
        copyfile(bagDescription["path"], dest )
        for add_file_key, add_file_val in additional_files.items():
            name_file = add_file_val.split('/')
            destadd = directory +"/"+name_file[len(name_file)-1]
            copyfile(add_file_val, destadd)
        
    except OSError as e:
        # directory already exists
        print (e)
    try:
        if config_rasm["add_in_data_base"] == True:
            insert_rasm_in_ddbb (name, description, directory, bagDescription["init_bag"], bagDescription["end_bag"], init_time, datetime.now()+timedelta(days=days_of_expiry), flag_expired, flag_max_size)
    except Exception as e:
        print ("I couldn't save in data base")
        print (e)
    


#Trigger cleaner
def callBackCleaner(event):        
     cleanerFunction()


def cleanerFunction(clear_temp_folder = False):
    if (clear_temp_folder == True):
        recursive_delete(config_rasm["temporal_bag_directory"])
    if (config_rasm["add_in_data_base"]==False and get_size(config_rasm["saved_bag_driectory"])>config_rasm["max_size"]):
        print ("The log size is : " + str(get_size(config_rasm["saved_bag_driectory"]))+ " MB")
        if(get_size(config_rasm["saved_bag_driectory"])>config_rasm["max_size"]):
           print ("Cleaning the directory: " +  config_rasm["saved_bag_driectory"])
           try:
               if (recursive_delete(config_rasm["saved_bag_driectory"])==True):
                    ddbb_update_rasm_delete_all()
           except Exception as e:
                print ("Error to try delete all files for max_size in saved_bag_directory")
                print (e)
    else:
        list_expired = ddbb_get_expired()
        for item in list_expired:
            try:
                if (recursive_delete(item, True)==True):
                    ddbb_update_expired(item)
            except Exception as e:
                print ("Error to try delete the file for time expired: " + item)
        
        if(get_size(config_rasm["saved_bag_driectory"])>config_rasm["max_size"]):
            list_max_size = ddbb_get_max_size()
            for item in list_max_size:
                try:
                    if (recursive_delete(item, True)==True):
                        ddbb_update_rasm_max_size(item)
                except Exception as e:
                    print ("Error to try delete the file for max_size: " + item)
        list_max_time_expired = ddbb_get_expired_max_time()
        for item in list_max_time_expired:
            try:
                if(recursive_delete(item, True)==True):
                    ddbb_update_expired(item)
            except Exception as e:
                print ("Error to try delete the file for max time expired: " + item)
        print ("The log size is : " + str(get_size(config_rasm["saved_bag_driectory"]))+ " MB")
        if(get_size(config_rasm["saved_bag_driectory"])>config_rasm["max_size"]):
           print ("Cleaning the directory: " +  config_rasm["saved_bag_driectory"])
           try:
               if (recursive_delete(config_rasm["saved_bag_driectory"])==True):
                    ddbb_update_rasm_delete_all()
           except Exception as e:
                print ("Error to try delete all files for max_size in saved_bag_directory")
                print (e)

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size*0.00000095367432 #to get Mb

def recursive_delete(start_path = None, delete_start_path = False):
    if(start_path==None):
        return False
    try:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                remove(os.path.join(dirpath, f))
        if delete_start_path==True:
            remove (start_path)
    except Exception as e:
        print (e)
        return False
    return True
 

bagDescription = {
    "path" :  None,
    "init_bag": None,
    "end_bag" : None
    
}



alarsm_rasm = {}


config_rasm = {
    'base_directory': '~/rasm_log/',
    'temporal_bag_directory': 'temp/',
    'saved_bag_driectory': 'saved',
    'frame_bag': 30,
    'max_size': 500, #MB
    'force_delet_files_days': 30,
    'add_in_data_base': False,
    'topics_exclude': []
}


conf_ddbb = {
    'user': "root",
    'password': "R0b0tn1K",
    'host': "127.0.0.1",
    'database': "db_astrobot"
    
}

topics_subscribe = None

rasm_logic = None

#Charge the yaml
def chargeYaml():
    global topics_subscribe
    topics_subscribe = None
    global rasm_logic 
    rasm_logic = None
    global conf_ddbb
    global config_rasm
    if rospy.has_param(yaml+'conf_ddbb/user'):
        conf_ddbb['user'] = rospy.get_param(yaml+'conf_ddbb/user')
        print("ddbb user: "+conf_ddbb['user'])
    if rospy.has_param(yaml+'conf_ddbb/password'):
        conf_ddbb['password'] = rospy.get_param(yaml+'conf_ddbb/password')
        print("ddbb password: "+conf_ddbb['password'])
    if rospy.has_param(yaml+'conf_ddbb/host'):
        conf_ddbb['host'] = rospy.get_param(yaml+'conf_ddbb/host')
        print("ddbb host: "+conf_ddbb['host'])
    if rospy.has_param(yaml+'conf_ddbb/database'):
        conf_ddbb['database'] = rospy.get_param(yaml+'/conf_ddbb/database')
        print("ddbb database: "+conf_ddbb['database'])
    if rospy.has_param(yaml+'/config_rasm/base_directory'):
        config_rasm['base_directory'] = rospy.get_param(yaml+'/config_rasm/base_directory')
        print("config rasm base_directory: "+config_rasm['base_directory'])
    if rospy.has_param(yaml+'/config_rasm/temporal_bag_directory'):
        config_rasm['temporal_bag_directory'] = rospy.get_param(yaml+'/config_rasm/temporal_bag_directory')
        print("config rasm temporal_bag_directory: "+config_rasm['temporal_bag_directory'])
    if rospy.has_param(yaml+'/config_rasm/saved_bag_driectory'):
        config_rasm['saved_bag_driectory'] = rospy.get_param(yaml+'/config_rasm/saved_bag_driectory')
        print("config rasm saved_bag_driectory: "+config_rasm['saved_bag_driectory'])
    if rospy.has_param(yaml+'/config_rasm/frame_bag'):
        config_rasm['frame_bag'] = rospy.get_param(yaml+'/config_rasm/frame_bag')
        print("config rasm frame_bag: "+str(config_rasm['frame_bag']))
    if rospy.has_param(yaml+'/config_rasm/max_size'):
        config_rasm['max_size'] = rospy.get_param(yaml+'/config_rasm/max_size')
        print("config rasm max_size: "+str(config_rasm['max_size']))
    if rospy.has_param(yaml+'/config_rasm/force_delet_files_days'):
        config_rasm['force_delet_files_days'] = rospy.get_param(yaml+'/config_rasm/force_delet_files_days')
        print("config rasm force_delet_files_days: "+str(config_rasm['force_delet_files_days']))
    if rospy.has_param(yaml+'/config_rasm/add_in_data_base'):
        config_rasm['add_in_data_base'] = rospy.get_param(yaml+'/config_rasm/add_in_data_base')
        print("config rasm add_in_data_base: "+str(config_rasm['add_in_data_base']))
    if rospy.has_param(yaml+'/config_rasm/topics_exclude'):
        config_rasm['topics_exclude'] = rospy.get_param(yaml+'/config_rasm/topics_exclude')
        print("config rasm topics_exclude: "+str(config_rasm['topics_exclude']))
    if rospy.has_param(yaml+'/topics_subscribe'):
        topics_subscribe = copy.deepcopy(rospy.get_param(yaml+'/topics_subscribe'))
        print("length topics: "+ str(len(topics_subscribe)))
    if rospy.has_param(yaml+'/rasm_logic'):
        rasm_logic = copy.deepcopy(rospy.get_param(yaml+'/rasm_logic'))
        print("length rasm_logic: "+ str(len(rasm_logic)))

struct_alarm_rasm = {
    'name': "",
    'description': "",
    'active': False,
    'init_time': None,
    'saved': True,
    'trigger': "",
    'flag_expired_time': True,
    'days_of_expiry': 1,
    'flag_erase_max_size_rasm': True,
    'topics_used': {},
    'not_init_in': 0,
    'not_new_msg_in': 0,
    'add_files': {}
    
}

rules_rasm = []

#Initialitalize the subscribe
def initSubscriber():
    global topics_subscribe
    if topics_subscribe == None:
        return False
    for topic_key, topic_val in topics_subscribe.items():
        generalSubscribe(topic_val["topic"], topic_key, topic_val["type_msg"], topic_key)
    return True

def chargeRuleRasm():
    global rules_rasm
    global rasm_logic
    global currentCallBacks
    global informationCallBacks
    #print(len(currentCallBacks))
    #for key, value in currentCallBacks.iteritems():
    #    print (key)
    #print(len(informationCallBacks))
    if rasm_logic == None:
        return False
    for rasm_logic_key, rasm_logic_val in rasm_logic.items():
        aux = copy.deepcopy(struct_alarm_rasm)
        aux["name"] = rasm_logic_key
        aux["description"] = rasm_logic_val["description"]
        if 'days_of_expiry' in rasm_logic_val:
            aux["days_of_expiry"] = rasm_logic_val["days_of_expiry"]
            if (rasm_logic_val["days_of_expiry"]>0):
                aux["flag_expired_time"] = True
            else:
                aux["flag_expired_time"] = False
        if 'flag_erase_max_size_rasm' in rasm_logic_val:
            aux["flag_erase_max_size_rasm"] = rasm_logic_val["flag_erase_max_size_rasm"]
        if 'not_init_in' in rasm_logic_val:
            aux["not_init_in"] = rasm_logic_val["not_init_in"]
        if 'not_new_msg_in' in rasm_logic_val:
            aux["not_new_msg_in"] = rasm_logic_val["not_new_msg_in"]
        if 'trigger' in rasm_logic_val:
            for topic in re.findall(r'!--(.*?)--!', rasm_logic_val["trigger"]):
                if (informationCallBacks[topic] == None):
                    print ("Not exist this topic: " + topic)
                    return False

                aux["topics_used"][topic] = currentCallBacks[topic]
            for topic in re.findall(r'/--(.*?)--/', rasm_logic_val["trigger"]):
                if (informationCallBacks[topic] == None):
                    print ("Not exist this topic: " + topic)
                    return False
                aux["topics_used"][topic] = currentCallBacks[topic]
            rasm_logic_val["trigger"] = rasm_logic_val["trigger"].replace("!--", " currentCallBacks['")
            rasm_logic_val["trigger"] = rasm_logic_val["trigger"].replace("--!", "']")
            rasm_logic_val["trigger"] = rasm_logic_val["trigger"].replace("/--", " informationCallBacks['")
            rasm_logic_val["trigger"] = rasm_logic_val["trigger"].replace("--/", "']")
            aux['trigger'] = rasm_logic_val["trigger"]
            
        else:
            if(aux["not_init_in"]!=False or aux["not_new_msg_in"]!=False):
                print ("It is neccesary a trigger")
                return False
            
        if 'add_files' in rasm_logic_val:
            for add_file_key, add_file_val in rasm_logic_val["add_files"].items():
                aux["add_files"][add_file_key] = add_file_val
        
        rules_rasm.append(copy.deepcopy(aux))
    return True



 #Create the directories   
def createDirectory():
    global config_rasm
    try:
        if not os.path.exists(config_rasm["base_directory"]):
            os.makedirs(config_rasm["base_directory"])
            print ("Created -> " + config_rasm["base_directory"])
        else:
            print ("The directory exist -> " + config_rasm["base_directory"])
    except OSError as e:
        # directory already exists
        print (e)
    config_rasm["temporal_bag_directory"] = config_rasm["base_directory"]+"/"+config_rasm["temporal_bag_directory"]
    try:
        if not os.path.exists(config_rasm["temporal_bag_directory"]):
            os.makedirs(config_rasm["temporal_bag_directory"])
            print ("Created -> " + config_rasm["temporal_bag_directory"])
        else:
            print ("The directory exist -> " + config_rasm["temporal_bag_directory"])
    except:
        # directory already exists
        print (e)
    config_rasm["saved_bag_driectory"] = config_rasm["base_directory"]+"/"+config_rasm["saved_bag_driectory"]
    try:
        if not os.path.exists(config_rasm["saved_bag_driectory"]):
            os.makedirs(config_rasm["saved_bag_driectory"])
            print ("Created -> " + config_rasm["saved_bag_driectory"])
        else:
            print ("The directory exist -> " + config_rasm["saved_bag_driectory"])
    except:
        # directory already exists
        print (e)
    
            
rospy.init_node('rasm', anonymous=True)

#read yaml:
yaml = '/rasm/'
chargeYaml()
createDirectory()
if (initSubscriber() == False):
    rospy.loginfo("Not defined any subscriber in yaml "+yaml+"/topics_subscribe")
    exit()
if (chargeRuleRasm() == False):
    exit()
#Trigger to start the bag
rospy.Timer(rospy.Duration(config_rasm["frame_bag"]+3), callBackTriggerBag)
#The logic of rasm
rospy.Timer(rospy.Duration(0.5), callBackTriggerLogic)
#The cleaner
rospy.Timer(rospy.Duration(600), callBackCleaner)
cleanerFunction(True)
rospy.spin()
'''
generalSubscribe("/test", "StingTest", "std_msgs/String", "TestString")
rospy.Timer(rospy.Duration(50), callBackTriggerLogic)
'''

'''    
while ("StringTest" in currentCallBacks)==False:
    None
for x in range(0, 10000):
    print (currentCallBacks["StringTest"].data)
'''
#insert_rasm_in_ddbb ("Test", "Test add alarm", "directory", datetime.now(), datetime.now()+timedelta(days=1), datetime.now())
