ip = 'raspberrypi' # Change to the IP or hostname of the Pi

vit = True # Use Vit model (True) or Mobilenet based model (False)
normalize_to_imnet = True

# checkpoint_path = "checkpoints/saved/epoch=4-step=1015.ckpt" # Path to the model checkpoint file
# checkpoint_path = "checkpoints/saved/epoch=25-step=5278.ckpt"
checkpoint_path = "checkpoints/epoch=9-step=2460.ckpt"



accelerator = "mps" # or "cpu", etc..


