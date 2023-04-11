import json
import time

import pandas as pd
import requests
import paho.mqtt.client as mqtt


def http_sender(map_data, decomposition_data, timeout):
    server = "http://0.0.0.0:8080/"
    m_lines = max(map_data.shape[0], decomposition_data.shape[0])
    for index in range(m_lines):
        if index < map_data.shape[0]:
            requests.post(f"{server}/map_position_insert", data=json.dumps(map_data.iloc[index].to_dict()))
        if index < csv_decomposition.shape[0]:
            requests.post(f"{server}/variable_decomposition_insert",
                          data=json.dumps(decomposition_data.iloc[index].to_dict()))
        # Wait timeout seconds
        time.sleep(timeout)


def mqtt_sender(map_data, decomposition_data, timeout):
    host, port = "0.0.0.0", 1883
    topic_map, topic_decomposition = "map_position_insert", "variable_decomposition_insert"
    client = mqtt.Client("simulator")
    client.connect(host=host, port=port)
    m_lines = max(map_data.shape[0], decomposition_data.shape[0])
    for index in range(m_lines):
        if index < map_data.shape[0]:
            client.publish(topic_map, json.dumps(map_data.iloc[index].to_dict()), qos=0, retain=False)
        if index < decomposition_data.shape[0]:
            client.publish(topic_decomposition, json.dumps(decomposition_data.iloc[index].to_dict()), qos=0, retain=False)
        # Wait timeout seconds
        time.sleep(0.1)
        print(f"Sent {index} lines")
    client.disconnect()


if __name__ == "__main__":
    h2_decomposition_filename = "./data/h2_decomposition_data.csv"
    map_data_filename = "./data/map_data.csv"
    timeout = .1
    csv_decomposition = pd.read_csv(h2_decomposition_filename)
    csv_map_data = pd.read_csv(map_data_filename)

    # commit the data to reset the state on the server
    requests.get("http://0.0.0.0:8080/commit")
    # start the simulation
    # http_sender(csv_map_data, csv_decomposition, timeout)
    mqtt_sender(csv_map_data, csv_decomposition, timeout)
