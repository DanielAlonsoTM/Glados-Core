import os


def establish_connection(ssdi, password):
    try:
        # Edit Wifi name and password for config file
        cmd = "wpa_passphrase " + ssdi + " " + password + " > /etc/wpa_supplicant.conf"
        os.system(cmd)

        # Make connection
        cmd = "wpa_supplicant -B -i interface -c /etc/wpa_supplicant.conf"
        os.system(cmd)
    except OSError as error:
        print(error)
