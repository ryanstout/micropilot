import socket
import cv2
from picarx import Picarx
import struct
import numpy as np
from time import sleep, time

# Initialize PiCar-X
px = Picarx()
px.forward(0)

# Set up the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# TCP Server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
host_ip = '0.0.0.0'  # Listen on all network interfaces
port = 12345
server_socket.bind((host_ip, port))
server_socket.listen(1)
print("Waiting for a connection.")
connection, addr = server_socket.accept()
print("Connected to", addr)

def send_frame(image):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame = cv2.imencode('.jpg', image, encode_param)
    data = frame.tobytes()
    connection.sendall(struct.pack("<L", len(data)) + data)

try:
    while True:

        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print(".")
            continue
        
        # Resize frame to 224x224
        frame = cv2.resize(frame, (224, 224))
    
        # Send frame to client
        send_frame(frame)

        # Receive steering and forward_value
        received_data = connection.recv(8)  # 2 floats, 4 bytes each
        if not received_data:
            break
        steering, forward_value = struct.unpack('ff', received_data)
        # print("Received - Steering:", steering, "Forward value:", forward_value)
        
        # Set steering and forward on PiCar-X
        px.set_dir_servo_angle((steering * 20) + (0.3 * 20))

        forward_value = 2
        px.forward(forward_value)

        sleep(0.05)
finally:
    cap.release()
    connection.close()
    server_socket.close()
