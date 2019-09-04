import os
import subprocess

# cmd_ls = 'ls -l /'
# os.system(cmd_ls)

cmd_nmap = 'nmap -sP 192.168.1.1/24'
status, result = subprocess.getstatusoutput(cmd_nmap)

if status == 0:
    print('status value = ' + str(status))
    # os.system(cmd_nmap)
else:
    print("nmap is not installed")
