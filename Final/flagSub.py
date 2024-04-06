import paho.mqtt.client as mqtt

def on_message(client, userdata, message):
    print(f"Received message '{message.payload.decode()}' on topic '{message.topic}'")

client = mqtt.Client("Flag viewer")
client.on_message = on_message
client.connect("localhost", 1883)
client.subscribe("flagged_results")
client.loop_forever()
