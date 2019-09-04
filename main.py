from module_packages.manager import check_package_installed
from module_network.scanning import *

cmd = "nmap"
check_package_installed(cmd)

check_hosts_network()
