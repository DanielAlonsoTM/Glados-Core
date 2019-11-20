import configparser


def read_config_ini(title, item):
    ini_file = "path_init_file"
    config = configparser.ConfigParser()

    config.sections()
    config.read(ini_file)
    config.sections()

    # config['DEFAULT']['item']
    return config[title][item]
