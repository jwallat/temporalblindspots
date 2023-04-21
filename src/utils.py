import pandas as pd
import json


def load_dataset(ds_path):
    data = pd.read_csv(ds_path)

    return data


def load_json(path):
    with open(path) as file:
        json_data = json.load(file)

    return json_data
