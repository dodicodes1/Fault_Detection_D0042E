import paho.mqtt.client as mqtt
import uuid #Unique user id
import json

N = 5

BROKER = "emaintlab-D0042E-broker.azurewebsites.net"
TOPICS = [
    "emaint/house1/sensor/temperature/room1",
    "emaint/house1/sensor/temperature/room2",
    "emaint/house1/sensor/temperature/room3",
    "emaint/house1/sensor/temperature/supply",
    "emaint/house1/sensor/temperature/outside",
    "emaint/house1/sensor/flow/room1",
    "emaint/house1/sensor/flow/room2",
    "emaint/house1/sensor/flow/room3",
    "emaint/house1/sensor/return/room1",
    "emaint/house1/sensor/return/room2",
    "emaint/house1/sensor/return/room3",
]

# Callback when connected
def on_connect(client, userdata, flags, rc, properties):
    print("Connected with result code", rc)
    for topic in TOPICS:
        client.subscribe(topic)

buffer = [[],[],[],[],[],[],[],[],[],[],[]]

def on_message(client, userdata, msg):
    value = json.loads(msg.payload)['value']
    topic = msg.topic
    topic_index = TOPICS.index(topic)
    #print(topic_index,topic,value)
    if len(buffer[topic_index]) < N:
        buffer[topic_index].append(value)
    else:
        buffer[topic_index] = buffer[topic_index][1:]+[value]


client_id = f"client-{uuid.uuid4()}"
client = mqtt.Client(
    client_id = client_id,
    transport="websockets",
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
)
username = "username" # set username
password = "password" # set password
client.username_pw_set(username,password)

client.on_connect = on_connect
client.on_message = on_message

client.tls_set()

client.connect(BROKER, 443, 60)

import torch.nn as nn
import torch.nn.functional as F
class FaultNet(nn.Module):
    def __init__(self, n_features, n_outputs):
        super().__init__() # Calls parent class (nn.Module) constructor
        # Creates a sequential model
        self.net = nn.Sequential(
            nn.Linear(n_features, 32), # Layer 1
            nn.ReLU(), # Activation function (non-linearity)
            nn.Linear(32, 16), # Layer 2
            nn.ReLU(),
            nn.Linear(16, n_outputs) # Output layer
        )
    # Defines how data flows through the model
    def forward(self, x):
        return self.net(x)
    
import torch
model = torch.load("model.pt", weights_only=False)
model.eval()
import pickle
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

import numpy as np
def main():
    X_all = np.array(buffer).T
    X_raw = X_all[-1:,:]
    X_sp_err = np.array([[21,22,20]]) - X_raw[:,0:3]
    X_P = X_raw[:,5:8]*(X_raw[:,3:4] - X_raw[:,-3:])
    X_mean = np.mean(X_all,axis=0,keepdims=True)
    #X_std = np.std(X_all,axis=0,keepdims=True)
    X = np.concatenate([X_raw, X_sp_err, X_P, X_mean], axis = 1)
    
    X_scaled = scaler.transform(X)
    features = torch.tensor(X_scaled,dtype=torch.float32)
    prediction = model(features)
    probabilities = F.softmax(prediction,dim=1).detach().numpy()
    print(f"Probability of no fault: {probabilities[0,0]*100:.2f} %")
    print(f"Probability of fault: {probabilities[0,1]*100:.2f} %")
    print()

import time
try:
    client.loop_start()
    time.sleep(5)
    while True:
        main()
        time.sleep(15)
finally:
    client.loop_stop()
    client.disconnect()

