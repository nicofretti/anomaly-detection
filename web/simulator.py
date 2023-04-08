import json
import time

import pandas as pd
import requests

if __name__ == "__main__":
    server_url = "http://0.0.0.0:8050/"
    h2_decomposition_filename = "./data/h2_decomposition_data.csv"
    map_data_filename = "./data/map_data.csv"
    timeout = .1
    csv_decomposition = pd.read_csv(h2_decomposition_filename)
    csv_map_data = pd.read_csv(map_data_filename)

    m_lines = max(csv_map_data.shape[0], csv_decomposition.shape[0])

    # commit the data to reset the state on the server
    requests.get(f"{server_url}/commit")

    for index in range(m_lines):
        if index < csv_map_data.shape[0]:
            requests.post(f"{server_url}/map_position_insert", data=json.dumps(csv_map_data.iloc[index].to_dict()))
        if index < csv_decomposition.shape[0]:
            requests.post(f"{server_url}/variable_decomposition_insert", data=json.dumps(csv_decomposition.iloc[index].to_dict()))
        # Wait timeout seconds
        time.sleep(timeout)

