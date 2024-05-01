import socket
import cv2
import os
import numpy as np
import struct
import time
from model.model import SteeringModel
import torchvision.transforms as transforms
from PIL import Image
from pynput import keyboard

# Connect to the server
while True:
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('192.168.1.212', 12345)  # Replace with the server IP
        client_socket.connect(server_address)
        print("Connection successful.")
        break
    except socket.error as ex:
        print("Connection failed. Retrying...")
        time.sleep(1)

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# Load model from pl.LightningModule
print("Load model")
model = SteeringModel.load_from_checkpoint(checkpoint_path="checkpoints/epoch=8-step=360.ckpt")
model.to('mps')
print("loaded")

normalize = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

try:
    steering = 0.0  # Default steering
    forward_value = 0.0  # Default forward value

    def on_press(key):
        global forward_value
        if key == keyboard.Key.up:  # if the 'up' arrow key is pressed
            print("Go")
            forward_value = 1.0
        else:
            print("stop")
            forward_value = 0.0

    # Start listening to keyboard
    # listener = keyboard.Listener(on_press=on_press)
    # listener.start()
    while True:
        # Receive frame
        length = struct.unpack("<L", recvall(client_socket, 4))[0]
        frame_data = recvall(client_socket, length)
        frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # Display frame
        cv2.imshow('Video', frame)

        # Convert the frame to a batch of size 1 tensor
        #  Convert the image from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the image from OpenCV to PIL format
        pil_image = Image.fromarray(frame)

        # Apply transformations
        tensor_image = normalize(pil_image)
        tensor_image = tensor_image.unsqueeze(0)

        # Run frame through pytorch lightning model
        preds = model(tensor_image.to('mps'))
        steering = preds[0].item()
        print("Steering: ", steering)

        # Check if space is pressed for forward motion
        key = cv2.waitKey(1) & 0xFF
        if key == ord('g'):
            print("Go")
            forward_value = 1.0
        else:
            print("stop")
            forward_value = 0.0

        if key == ord('q'):
            break
        
        # Send steering and forward_value to the server
        client_socket.sendall(struct.pack('ff', steering, forward_value))

finally:
    cv2.destroyAllWindows()
    client_socket.close()