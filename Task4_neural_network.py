import paho.mqtt.client as mqtt
import uuid #Unique user id
import json
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pickle
import time

# Number of recent samples stored for each sensor
N = 5

# MQTT broker address
BROKER = "emaintlab-D0042E-broker.azurewebsites.net"
# List of subscribed sensor topics
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

# Called when connected to the broker
def on_connect(client, userdata, flags, rc, properties):
    print("Connected with result code", rc)
    # Subscribe to all topics
    for topic in TOPICS:
        client.subscribe(topic)

# Buffer storing recent sensor values
buffer = [[],[],[],[],[],[],[],[],[],[],[]]

# Called when a new MQTT message is received
def on_message(client, userdata, msg):
    # Extract sensor value from JSON payload
    value = json.loads(msg.payload)['value']
    topic = msg.topic
    topic_index = TOPICS.index(topic)

    # Store latest N values
    if len(buffer[topic_index]) < N:
        buffer[topic_index].append(value)
    else:
        buffer[topic_index] = buffer[topic_index][1:]+[value]

# Generate unique client ID
client_id = f"client-{uuid.uuid4()}"

# Create MQTT client
client = mqtt.Client(
    client_id = client_id,
    transport="websockets",
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
)
username = "username" # set username
password = "password" # set password
client.username_pw_set(username,password)

# Assign callback functions
client.on_connect = on_connect
client.on_message = on_message

# Enable TLS encryption
client.tls_set()

# Connect to broker
client.connect(BROKER, 443, 60)

# Neural network model
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
    
# Load trained model
model = torch.load("model.pt", weights_only=False)
model.eval()

# Load feature scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def main():
    # Convert buffer to NumPy array
    X_all = np.array(buffer).T

    # Latest sensor sample
    X_raw = X_all[-1:,:]

    # Temperature setpoint errors
    X_sp_err = np.array([[21,22,20]]) - X_raw[:,0:3]

    # Estimated radiator power
    X_P = X_raw[:,5:8]*(X_raw[:,3:4] - X_raw[:,-3:])
    
    # Rolling mean values
    X_mean = np.mean(X_all,axis=0,keepdims=True)
    
    # Combine all features
    X = np.concatenate([X_raw, X_sp_err, X_P, X_mean], axis = 1)
    
    # Scale features
    X_scaled = scaler.transform(X)

    # Convert to tensor
    features = torch.tensor(X_scaled,dtype=torch.float32)

    # Model prediction
    prediction = model(features)

    # Convert output to probabilities
    probabilities = F.softmax(prediction,dim=1).detach().numpy()

    print(f"Probability of no fault: {probabilities[0,0]*100:.2f} %")
    print(f"Probability of fault: {probabilities[0,1]*100:.2f} %")
    print()

try:
    # Start MQTT loop
    client.loop_start()
    time.sleep(5)
    while True:
        # Run fault detection
        main()
        time.sleep(15)
finally:
    # Stop MQTT client
    client.loop_stop()
    client.disconnect()

