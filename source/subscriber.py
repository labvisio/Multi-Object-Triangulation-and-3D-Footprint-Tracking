from __future__ import print_function
from is_wire.core import Channel, Subscription

from protobuf.message_pb2 import  Detections, Points
# Connect to the broker
channel = Channel("amqp://guest:guest@10.20.5.3:30000")

# Subscribe to the desired topic(s)
subscription = Subscription(channel)
subscription.subscribe(topic="is.tracker.detections")
# ... subscription.subscribe(topic="Other.Topic")

# Blocks forever waiting for one message from any subscription
while True:
    # Consume a message from the channel
    message = channel.consume(timeout=5.0)  # Adjust timeout as needed
    if message is None:
        print("No message received within the timeout period.")
        continue

    # Unpack the message to get the Detection object
    detection = message.unpack(Detections)

    # Process the detection (for example, print it)
    # print(f"Received detection: {detection}")
    print("Timestamp:", detection.timestamp)    
    print("Frame:", detection.frame)
    print("3D Points:")
    for point in detection.points:
        print(f"ID: {point.id}, Name: {point.name}, Coordinates: ({point.position[0]}, {point.position[1]}, {point.position[2]})")
    print("-\n"*5)

    # Process the points in the detection
    