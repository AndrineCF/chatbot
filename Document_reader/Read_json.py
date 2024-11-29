import json

def read_json_file(filepath):
    data = None

    print("Reading json file")
    with open(filepath) as f:
        data = json.load(f)
    print("Done reading json file")

    return data