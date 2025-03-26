# from: /Users/jinyuj/Workspace/PythonProjects/FirstPython/src/Tests/csv2json

import csv
import json


# Function to convert a CSV to JSON
# Takes the file paths as arguments
def parse_csv_file_to_json(path_file_csv):
    # create a dictionary
    elements = []

    # Open a csv reader called DictReader
    with open(path_file_csv, encoding='utf-8') as file_csv:
    #with open(path_file_csv) as file_csv:
        reader_csv = csv.DictReader(file_csv)

        # Convert each row into a dictionary
        # and add it to data
        for dict_head_value in reader_csv:
            element = {}

            for head, value in dict_head_value.items():
                #print(value)
                if value and (value[0] in ["[", "{"]):
                    #element[head] = eval(value)
                    element[head] = value
                else:
                    element[head] = value

            elements.append(element)
        # end
    # end

    return elements
# end
