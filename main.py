import os

cmd_ls = 'ls -l /'
cmd_nmap = 'nmap -sP 192.168.1.1/24'

os.system(cmd_ls)
os.system(cmd_nmap)
