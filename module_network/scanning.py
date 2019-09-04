import os
import subprocess

cmd_nmap = "nmap"


def check_hosts_network():
    try:
        default_gateway = get_default_gateway()
        network_mask = get_current_subnet_mask()

        os.system(cmd_nmap + " -sP " + default_gateway + "/" + network_mask)
    except AssertionError as error:
        print(error)


def get_interface_name():
    return subprocess.getoutput("ifconfig -s| grep wl | cut -d ' ' -f 1")


def get_current_ip():
    return subprocess.getoutput("hostname -I")


def get_current_network_mask():
    interface_name = get_interface_name()
    return subprocess.getoutput("ifconfig" + interface_name + "| grep -Po 'netmask \K.*' | cut -d ' ' -f 1")


def get_current_subnet_mask():
    return subprocess.getoutput("ip -o -f inet addr show | awk '/scope global/ {print $4}' | cut -d '/' -f 2")


def get_default_gateway():
    return subprocess.getoutput("ip r | grep default | cut -d ' ' -f 3")
