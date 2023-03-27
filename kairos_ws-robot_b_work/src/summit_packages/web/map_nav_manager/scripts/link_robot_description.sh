#!/bin/sh
if [ $# -eq 1 ] 
	then
		echo Program: $0. Param: $1
		ln -sf `rospack find $1` `rospack find map_nav_manager`'/web/'$1 
else
	echo "$0: Incorrect number of parameters: the name of the ros package description is needed"
fi

