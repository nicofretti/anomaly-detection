#!/bin/bash
source ~/kairos_ws/devel/setup.bash && roscd robot_bringup
sudo cp controller/50-ds4drv.rules /etc/udev/rules.d/
sudo cp controller/ds4drv.service /etc/systemd/system/
sudo cp controller/ds4drv.conf /etc/

sudo systemctl enable ds4drv.service
sudo systemctl start ds4drv.service

