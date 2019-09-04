from module_packages.utilities import check_package_installed, install_package
from module_network.scanning import check_hosts_network

cmd = "nmap"
if check_package_installed(cmd):
    print("Package is installed")
else:
    print("Package is not installed")
    # install_package(cmd)

check_hosts_network()
