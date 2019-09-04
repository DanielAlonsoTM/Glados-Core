import os
import subprocess


# This method will be fix, and not show any result post execution of check 'package_name'
def check_package_installed(package_name):
    try:
        subprocess.call([package_name])
        return True
    except OSError as error:
        print(error)
        return False


def install_package(package_name):
    sudo_password = 'password'
    command = 'apt install ' + package_name
    os.system('echo %s|sudo -S %s' % (sudo_password, command))
