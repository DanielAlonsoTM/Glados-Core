import os
from utils.read_ini import read_config_ini


def establish_connection(ssdi, password):
    try:
        cmd_wpa_passphrase = 'wpa_passphrase "{}" "{}"  >> /etc/wpa_supplicant/wpa_supplicant.conf'.format(ssdi, password)

        # Get user credential
        key = read_config_ini('DEFAULT', 'password')

        cmd_wpa_passphrase = "echo {} | sudo -S sh -c '{}'".format(key, cmd_wpa_passphrase)

        # Add new SSID to /etc/wpa_supplicant/wpa_supplicant.conf
        os.system(cmd_wpa_passphrase)

        # Make connection
        # cmd_wpa_supplicant = "wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf"
        cmd_wpa_supplicant = "echo {} | sudo -S wpa_cli -i wlan0 reconfigure".format(key)
        os.system(cmd_wpa_supplicant)

    except OSError as error:
        print(error)
