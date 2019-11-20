import os.path


def check_file_or_path(path):
    if os.path.exists(path):
        return True
    else:
        return False
