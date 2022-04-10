import json


# Write a dictionary to json formatted file
def write_dictionary_json(dict_, output_path, mode='w'):
    with open(output_path, mode) as outfile:
        json.dump(dict_, outfile)


# Write a list to a txt file
def write_list_txt(content, output_path, mode='w'):
    with open(output_path, mode) as outfile:
        for item in content:
            outfile.write(item + '\n')
