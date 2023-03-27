#!/bin/bash
#
# Description:   ds4drv installer
#
# Company:       Robotnik Automation S.L.L.
# Creation Year: 2021
# Author:        Guillem Gari <ggari@robotnik.es>
#
#
# Copyright (c) 2021, Robotnik Automation S.L.L.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Robotnik Automation S.L.L. nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Robotnik Automation S.L.L. BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#Colour
red_colour='\033[0;31m'
green_colour='\033[0;32m'
light_purple_colour='\033[1;35m'
err_colour="${red_colour}"
nfo_colour="${light_purple_colour}"
suc_colour="${green_colour}"
no_colour='\033[0m'

#root permision

err_str_root_permission="You need root privileges try:\nsudo ${0}"

#udev
udev_rule_file="55-ds4drv.rules"
udev_rule_destiny="/etc/udev/rules.d/${udev_rule_file}"

udev_rule_copy_command="cp ${udev_rule_file} ${udev_rule_destiny}"
udev_reload_rules_command="udevadm control --reload-rules && udevadm trigger"

suc_str_udev_install='udev rules active'

err_str_udev_copy="error copying udev rules ${udev_rule_file}"
err_str_udev_reload="error reloading udev rules"

nfo_str_udev_install='Installing udev rules'

#systemd
systemd_service_file="ds4drv.service"
systemd_service_destiny="/etc/systemd/system/${systemd_service_file}"

systemd_service_copy_command="cp ${systemd_service_file} ${systemd_service_destiny}"
systemd_daemon_reload_command="systemctl daemon-reload"
systemd_enable_service_command="systemctl enable ${systemd_service_file}"
systemd_start_service_command="systemctl enable ${systemd_service_file}"

suc_str_systemd_install='systemd service active'

err_str_systemd_copy="error copying systemd service ${udev_rule_file}"
err_str_systemd_reload="error systemd reloading daemon"
err_str_systemd_enable="error enabling ${systemd_service_file} systemd service"
err_str_systemd_start="error starting ${systemd_service_file} systemd service"

nfo_str_systemd_install='installing service active'

suc_str_ds4drv_install='ds4drv installed'

ds4drv_search_python_command='which python'
ds4drv_search_python3_command='which python3'
ds4drv_search_pip_command='${python_bin} -m pip --version &>/dev/null'
ds4drv_pip_install_ds4dr_command='${python_bin} -m pip install ds4drv'

err_str_ds4drv_search_python='python not found'
err_str_ds4drv_search_pip='pip in not installed'
err_str_ds4drv_pip_install_ds4dr='could install ds4drv'

nfo_str_ds4drv_install='installing ds4drv'

function print_error() {
	local message="${1}"
	eval "echo -e "'"'"${err_colour}[ERROR]${no_colour}:   ${message}"'"'" 2>&1"
}

function print_info() {
	local message="${1}"
	eval "echo -e "'"'"${nfo_colour}[INFO]${no_colour}:    ${message}"'"'""
}

function print_success() {
	local message="${1}"
	eval "echo -e "'"'"${suc_colour}[SUCCESS]${no_colour}: ${message}"'"'""
}

function check_root_permission() {
	if [[ "${EUID}" = 0 ]]; then
		return 0
	else
		print_error "${err_str_root_permission}"
		return 1
	fi
}

function copy_udev_rules() {
	if ! eval "${udev_rule_copy_command}"; then
		print_error "${err_str_udev_copy}"
		return 1
	fi
	return 0
}

function reload_udev_rules() {
	if ! eval "${udev_reload_rules_command}"; then
		print_error "${err_str_udev_reload}"
		return 1
	fi

	return 0
}

function copy_systemd_service() {
	if ! eval "${systemd_service_copy_command}"; then
		print_error "${err_str_systemd_copy}"
	fi
	return 0
}

function reload_systemd_daemons() {
	if ! eval "${systemd_daemon_reload_command}"; then
		print_error "${err_str_systemd_reload}"
	fi
	return 0
}

function enable_systemd_service() {
	if ! eval "${systemd_enable_service_command}"; then
		print_error "${err_str_systemd_enable}"
	fi
	return 0
}

function start_systemd_service() {
	if ! eval "${systemd_start_service_command}"; then
		print_error "${err_str_systemd_start}"
	fi
	return 0
}

function search_python3() {
	if eval "${ds4drv_search_python3_command}"; then
		python_bin="$(eval "${ds4drv_search_python3_command}")"
		return 0
	fi
	if ! eval "${ds4drv_search_python_command}"; then
		print_error "${err_str_ds4drv_search_python}"
		return 1
	fi
	python_bin="$(eval "${ds4drv_search_python_command}")"
	return 0

}
function search_pip() {
	if ! eval "${ds4drv_search_pip_command}"; then
		print_error "${err_str_ds4drv_search_pip}"
		return 1
	fi
}
function pip_install_ds4dr() {
	if ! eval "${ds4drv_pip_install_ds4dr_command}"; then
		print_error "${err_str_ds4drv_pip_install_ds4dr}"
		return 1
	fi
}

function install_ds4drv() {
	print_info "${nfo_str_ds4drv_install}"
	if ! search_python3; then
		return 1
	fi
	if ! search_pip; then
		return 1
	fi
	if ! pip_install_ds4dr; then
		return 1
	fi
	print_success "${suc_str_ds4drv_install}"
	return 0
}

function install_udev_rules() {
	print_info "${nfo_str_udev_install}"
	if ! copy_udev_rules; then
		return 1
	fi
	if ! reload_udev_rules; then
		return 1
	fi
	print_success "${suc_str_udev_install}"
	return 0
}

function install_systemd_service() {
	print_info "${nfo_str_systemd_install}"
	if ! copy_systemd_service; then
		return 1
	fi
	if ! reload_systemd_daemons; then
		return 1
	fi
	if ! enable_systemd_service; then
		return 1
	fi
	if ! start_systemd_service; then
		return 1
	fi
	print_success "${suc_str_systemd_install}"
	return 0
}


function main() {
	if ! check_root_permission; then
		return 1
	fi
	if ! install_ds4drv; then
		return 1
	fi
	if ! install_udev_rules; then
		return 1
	fi
	if ! install_systemd_service; then
		return 1
	fi
}

main "${@}"
exit $?
