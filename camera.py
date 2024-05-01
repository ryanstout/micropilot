from picamera2 import Picamera2, Preview

# Initialize Picamera2
picam2 = Picamera2()

# Set the configuration for 1080p resolution
config = picam2.create_preview_configuration(main={"size": (1280, 720)})
# config = picam2.create_preview_configuration(main={"size": (1920, 1080)})
picam2.configure(config)

# Start the camera
# picam2.start()

# Create a preview and display it (optional)
picam2.start_preview(Preview.QTGL)

# Capture an image
image = picam2.capture_array()

# Stop the preview
picam2.stop_preview()

# Stop the camera
# picam2.stop()
