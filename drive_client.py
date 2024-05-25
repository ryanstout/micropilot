import socket
import cv2
import os
import numpy as np
import struct
import time
from model.model import SteeringModel
import torchvision.transforms as transforms
from PIL import Image

# Connect to the server
while True:
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('100.107.241.102', 12345)  # Replace with the server IP
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
# model = SteeringModel.load_from_checkpoint(checkpoint_path="checkpoints/epoch=14-step=1110.ckpt")
# model = SteeringModel.load_from_checkpoint(checkpoint_path="checkpoints/epoch=9-step=1590.ckpt")
# model = SteeringModel.load_from_checkpoint(checkpoint_path="checkpoints/epoch=9-step=1700.ckpt")
# model = SteeringModel.load_from_checkpoint(checkpoint_path="checkpoints/epoch=34-step=5950.ckpt") # vit
# model = SteeringModel.load_from_checkpoint(checkpoint_path="checkpoints/epoch=1-step=406.ckpt") # vit
model = SteeringModel.load_from_checkpoint(checkpoint_path="checkpoints/epoch=4-step=1015.ckpt") # vit




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

        # steering = min(-1.0, max(1.0, steering * 1.5))
        print("Steering: ", steering, "Forward: ", forward_value)

        # Check if space is pressed for forward motion
        forward_value = 5.0
        # key = cv2.waitKey(5) & 0xFF
        # if key == ord('g'):
        #     forward_value = 1.0
        # else:
        #     forward_value = 0.0

        # if key == ord('q'):
        #     break
        
        # Send steering and forward_value to the server
        client_socket.sendall(struct.pack('ff', steering, forward_value))

finally:
    cv2.destroyAllWindows()
    client_socket.close()
