from module_packages.utilities import check_package_installed, install_package
from module_network.wifi_connecting import establish_connection

# from utils.read_ini import read_config_ini
# from module_network.scanning import check_hosts_network
# import os

cmd = "nmap"
if check_package_installed(cmd):
    print("Package is installed")
else:
    print("Package is not installed")
    install_package(cmd)

# check_hosts_network()

establish_connection("BSSID", "password")
