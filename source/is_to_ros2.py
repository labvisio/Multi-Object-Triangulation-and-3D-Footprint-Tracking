from protobuf.ros_pb2 import ROSMessage
from google.protobuf.struct_pb2 import Struct
from is_wire.core import Channel, Message



class SkeletonPosition:
    def __init__(self, channel, topic):
        self._channel = Channel(channel)
        self.topic = topic

    def get_ros_message(self, msg):
        ros_message = ROSMessage()
        ros_message.type = "std_msgs/msg/String"
        msg_dict = {
            'data': msg
        }
        string_msg = Struct()
        string_msg.update(msg_dict)
        ros_message = ROSMessage(content=string_msg)
        ros_message.type = "std_msgs/msg/String"

        return ros_message

    def send_to(self, msg):
        print(msg)
        ros_message = self.get_ros_message(msg)
        message = Message(content=ros_message)
        print(message)
        print(self._channel)
        self._channel.publish(message, topic=self.topic)