from is_msgs.image_pb2 import Image
from google.protobuf.json_format import Parse
from is_project.conf.options_pb2 import ServiceOptions
from is_wire.core import Channel, Message, Subscription, ContentType
from google.protobuf.message import Message as PbMessage
from google.protobuf.struct_pb2 import Struct
import socket
from typing import Tuple, Dict
import cv2 
import numpy as np
from protobuf.message_pb2 import Detections, Points
from datetime import datetime
# StreamChannel class for live camera feed
class StreamChannel(Channel):
    def __init__(
        self, uri: str = "amqp://guest:guest@localhost:5672", exchange: str = "is"
    ) -> None:
        super().__init__(uri=uri, exchange=exchange)

    def consume_last(self) -> Tuple[Message, int]:
        """
        Consume the last available message from the channel.
        """
        dropped = 0
        msg = super().consume()
        while True:
            try:
                # will raise an exception when no message remained
                msg = super().consume(timeout=0.0)
                dropped += 1
            except socket.timeout:
                return (msg, dropped)

# Function to convert Protocol Buffer Image to NumPy array
def to_np(image: Image) -> np.ndarray:
    """
    Convert a Protocol Buffer Image message to a NumPy array.
    """
    buffer = np.frombuffer(image.data, dtype=np.uint8)
    output = cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    return output

# Function to load JSON data
def load_json(filename: str, schema: PbMessage) -> PbMessage:
    """
    Load data from a JSON file and parse it into a Protocol Buffer message.
    """
    with open(file=filename, mode="r", encoding="utf-8") as f:
        proto = Parse(f.read(), schema())
    return proto

def publish(channel: Channel, frame:int, point_3d_list:list, track_ids:list, class_ids:list=None) -> None:
    """
    Publish data to a specified topic.
    
    Args:
        channel: The channel to publish to
        frame: Frame number
        point_3d_list: List of 3D points
        track_ids: List of track IDs
        class_ids: List of class IDs (optional)
    """
    detection = Detections()
    detection.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    detection.frame = frame

    for i, point in enumerate(point_3d_list):
        points = Points()

        # Usar a mesma lógica do save_3d_coordinates_with_ids
        points.position.extend(point)
        
        # Garantir ID válido com fallback (igual ao JSON)
        track_id = track_ids[i] if i < len(track_ids) else i
        points.id = int(track_id)
        
        # Garantir class válido com fallback (igual ao JSON)
        class_id = class_ids[i] if class_ids and i < len(class_ids) else None
        points.name = int(class_id) if class_id is not None else 1
        print(f"Sending: id={points.id}, name={points.name}, class_id_original={class_id}")

        detection.points.append(points)

    message = Message()
    message.content_type = ContentType.PROTOBUF
    message.pack(detection)
    channel.publish(message, topic="is.tracker.detections")