import os
from module_network.utilities import *

cmd_nmap = "nmap"


def check_hosts_network():
    try:
        default_gateway = get_default_gateway()
        network_mask = get_current_subnet_mask()

        os.system(cmd_nmap + " -sP " + default_gateway + "/" + network_mask)
    except AssertionError as error:
        print(error)
