import socket
import cv2
import pygame
from picarx import Picarx
import struct
import numpy as np
from time import sleep, time

# Initialize PiCar-X
px = Picarx()
px.forward(0)

# Initialize Pygame for Joystick
pygame.init()
pygame.joystick.init()

# Initialize the joystick
try:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Joystick initialized")
except pygame.error:
    print("No joystick found.")

# Set up the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# TCP Server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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

recording = False  # Initial state of recording

try:
    running = True

    # Pygame event handling
    steering = 0.0
    forward_value = 0.0
    while running:
        start_time = time()
        
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Resize frame to 720p
        frame = cv2.resize(frame, (1280, 720))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYAXISMOTION:
                if event.axis == 3:
                    steering = event.value
                    if steering > 1.0:
                        steering = 1.0
                    elif steering < -1.0:
                        steering = -1.0
                    
                    print("Steering:", steering)
                    px.set_dir_servo_angle((steering * 20) + (0.3 * 20))
                elif event.axis == 1:
                    forward_value = event.value
                    if forward_value < 0.1 and forward_value > -0.1:
                        forward_value = 0.0
                    forward_value *= -5.0
                    print("Forward value:", forward_value)
                    px.forward(forward_value)
            elif event.type == pygame.JOYBUTTONDOWN and event.button == 4:
                recording = True
            elif event.type == pygame.JOYBUTTONUP and event.button == 4:
                recording = False
        
        # Send data to client
        connection.sendall(struct.pack("fff", steering, forward_value, float(recording)))
        send_frame(frame)

        # Maintain a steady loop interval
        elapsed = time() - start_time
        sleep(max(0.1 - elapsed, 0))
finally:
    pygame.quit()
    cap.release()
    connection.close()
    server_socket.close()
