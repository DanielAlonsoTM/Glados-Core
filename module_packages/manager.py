import os
import subprocess


def check_package_installed(package_name):
    status, result = subprocess.getstatusoutput(package_name)

    if status == 0:
        os.system(package_name)
    else:
        print("package is not installed")


# ON CONSTRUCTION
def install_package(package_name):
    print("This function don't work yet")
    # os.system("sudo apt install " + package_name + " -y")
    # subprocess.call("sudo apt-get update", shell=True)
