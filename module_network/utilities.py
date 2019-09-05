import subprocess


def get_current_ip():
    return subprocess.getoutput("hostname -I")


def get_current_network_mask():
    interface_name = subprocess.getoutput("ifconfig -s | grep wl | cut -d ' ' -f 1")
    return subprocess.getoutput("ifconfig" + interface_name + "| grep -Po 'netmask \K.*' | cut -d ' ' -f 1")


def get_current_subnet_mask():
    return subprocess.getoutput("ip -o -f inet addr show | awk '/scope global/ {print $4}' | cut -d '/' -f 2")


def get_default_gateway():
    return subprocess.getoutput("ip r | grep default | cut -d ' ' -f 3")
