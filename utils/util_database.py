from random import randrange
from pymongo import MongoClient

CLIENT = MongoClient("127.0.0.1:27017")
DB = CLIENT['glados_db']


def get_instruction_collection(device_id):
    # Get collection
    instruction_collection = DB['instructions']
    # Get all instructions registers
    query = instruction_collection.find(
        {'contentInstruction.deviceId': device_id},  # Get only specific device items
        {'_id': 0, '_class': 0})  # Suppress _id and _class fields

    list_items = []

    for item_query in query:
        default_value = 0
        if item_query['contentInstruction']['action'] == 'TURN_ON':
            default_value = 1

        item_array = [default_value,
                      item_query['date'][11:-7],
                      # item['contentInstruction']['timeToExecute'],
                      randrange(6),
                      item_query['contentInstruction']['action']]

        list_items.append(item_array)

    return list_items
