#!/bin/sh 

# Run the mini-httpd webserver on the port defined in the
# mini-httpd.conf file and set the document directory to the current
# directory

PID=`pidof mini-httpd`

if [ ! -z $PID ]; then
    kill $PID
fi

sleep 1

echo "Launching mini-httpd..."

cd `rospack find map_nav_manager`/web
python -m SimpleHTTPServer 9001
#/usr/sbin/mini-httpd -C `rospack find map_nav_manager`/scripts/mini-httpd.conf -d `rospack find map_nav_manager`
