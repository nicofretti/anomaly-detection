import paho.mqtt.client as mqtt

if __name__ == '__main__':
    MQTT_CLIENT = mqtt.Client("kairos")
    MQTT_CLIENT.connect(host='157.27.184.45', port=1883)