import json


# Write a dictionary to json formatted file
def write_dictionary_json(dict_, output_path, mode='w'):
    with open(output_path, mode) as outfile:
        json.dump(dict_, outfile)
