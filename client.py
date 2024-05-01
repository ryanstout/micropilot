import socket
from time import time
import cv2
import os
import numpy as np
import struct

# Connect to the server
import socket
import time

# Next folder
folder_name = 0
while True:
    folder_name += 1
    # check if data/{i} exists
    if not os.path.exists(f"data/{folder_name}"):
        os.makedirs(f"data/{folder_name}")
        break


while True:
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('192.168.1.212', 12345)  # Replace with the server IP
        client_socket.connect(server_address)
        print("Connection successful.")
        break
    except socket.error as ex:
        print("Connection failed. Retrying...")
        time.sleep(1)  # Wait for a second before retrying


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

try:
    while True:
        # Receive float values
        data = client_socket.recv(12)  # 3 floats, 4 bytes each
        if not data:
            break
        steering, forward_value, recording = struct.unpack('fff', data) #
        print("Steering:", steering, "Forward value:", forward_value, "Recording:", recording)
        
        # Receive frame
        length = struct.unpack("<L", recvall(client_socket, 4))[0]
        frame_data = recvall(client_socket, length)
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        if recording and forward_value != 0.0:
            # Write the frame as a jpg to the data folder
            unix_timestamp = int(time.time())
            cv2.imwrite(f"data/{folder_name}/frame_{unix_timestamp}_{round(steering, 3)}_{round(forward_value, 3)}.jpg", frame)
        
        # Display frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    client_socket.close()
