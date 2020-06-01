import csv
from random import randrange


def get_all_times():
    list_times = []

    for hour in range(0, 24):
        for minute in range(0, 60):
            str_hour = ''
            str_minute = ''

            if hour < 10:
                str_hour = '0'
            if minute < 10:
                str_minute = '0'

            str_hour = str_hour + str(hour)
            str_minute = str_minute + str(minute)

            list_times.append('{}:{}'.format(str_hour, str_minute))

    return list_times


def create_csv(data, path_file):
    with open(path_file, 'w', newline='') as file_csv:
        wr = csv.writer(file_csv, delimiter=',')
        wr.writerow(['status', 'date', 'delay', 'action'])
        wr.writerows(data)


def create_test_data():
    list_time = get_all_times()

    list_data_dump = []

    action_string = ''

    for time in list_time:
        action_type = randrange(2)
        if action_type == 0:
            action_string = 'TURN_OFF'
        elif action_type == 1:
            action_string = 'TURN_ON'

        array_data = [action_type, time, randrange(6), action_string]
        list_data_dump.append(array_data)

    return list_data_dump
