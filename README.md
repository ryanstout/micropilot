# Self Driving (Toy) Car

MicroPilot is an autopilot for a DIY self driving (Toy) car, originally built during the Stem Club at [St. John Paul the Great Catholic Highschool](https://www.jpthegreatdenver.org/).

Put down masking tape and you can teach it to stay in the lines.

## How It Works

MicroPilot takes in frames from a off the shelf webcam and predicts the steering angle.

We created a course by putting down masking tape for lanes. The goal was to get it to follow the track and stay in the lanes using only the webcam footage as input.

It's probably possible to run these models on the Pi itself (especially the Pi5), but since there's no cooling, the Pi gets thermal throttled quite a bit. I found it easier to just connect the Pi to the Wifi, then record to and run inference on my machine. (Sending frames over a TCP connection)

To train the car, you attach a USB Joystick to the back and drive the course. When the left front button is held down, it records frames from the webcam and the steering angle of the joystick. The goal is to get a few types of training data:

- Footage driving the center of the track
- Footage of the car correcting when its pointing the wrong way. (So make the car face more towards the tape, then point the wheels hard in the correcting direction) Also have the car slightly go over the lane and record data of it correcting back into the lane.

The trick is to make sure the wheel direction changes smoothly. It can be tempting to drive by toggling between strong adjustments and straight. For good training data, you'll need to make your steering adjustments smooth. As you get closer to the edge, make your adjustments strong back towards the center of the lane. The training data should convey that when you're in the middle you need little adjustments and as you get closer to the tape you need stronger adjustments.

Once we collected training data, we built a model using the features from a pretrained Vision Transformer (Vit). The features from the Vit are fed into a small model that uses a series of Linear layers to predict the steering angle. Finally a Tanh layer constraints the output to between -1 and 1.



## The Hardware

It should work on any hardware, but v1 was built with thea RaspberryPi, an off the shelf webcam and 

- [SunFounder Smart Kit](https://www.amazon.com/gp/product/B0CGLPF29H/) - A car kit with most everything you need. While this makes it easy to get started with, the car itself is very slow and the steering has a lot of play in it and top speed is ridiculously slow. Were I to start over I would explore other options or building something DIY.
- 


# How to Train

To simplify things I used tailscale to map the Pi into my network. The client code has an `ip` variable you can change.

### Setup

I use poetry for the client and the server, but PiCarX needs to be installed manually since it's not on pip:

Follow the directions at https://github.com/sunfounder/picar-x

Then `poetry install` on the client (your machine) and the server (the Pi)

### Collecting Training Data

Plug the USB Joystick into the car. Then make sure you've set `ip` in train_client.py, then run `python train_server.py` on the Pi and `python train_client.py` on your machine. (I used deploy.sh to rsync the project to the Pi)

Hold down the top left button to start recording. Make sure your steering angles are correct when recording. See the "How it Works" section for details on what good training data looks like.

### Training the Model

Each time you run the train_client.py, it creates a new folder in data/. The folders will contain the webcam frame, and the filename will include the recorded steering angle.

SteeringDataModule in model.py takes in a list of directories with training. Update the paths on line 238 to the training folders you want to use. Then start training with:

```
python model.py
```

This will take some time :-) We used PyTorch Lightning for the model and the excellent `timm` library for the pretrained Vit. There is also a version based on mobilenetv3 (which is a much smaller model, but it's not quite as good) Change the `vit = True` if you want to try the mobilenet version. You could definitely run the mobilenet version on the car itself. (And maybe the Vit also)

As the model trains, it will save the top epochs in the checkpoints folder. I also have it setup to log to wandb, but this is optional. Once you have a top performing checkpoint, you can update the model path on line 36 of `drive_client.py`

### Run Autopilot

If you're not on a Apple Silicon Mac, be sure to change your accelerator on line 273 of `model.py`. To run the autopilot, put the car on the track and run `python drive_server.py` on the Pi, and `python drive_client.py` on your machine.

# Summary

Best of luck! This was a fun project that the students seemed really excited about. Due to limited time, we had to gloss over quite a few details, but I think this did a good job of getting them excited about computer science, engineering, and AI. It's also great how approachable AI has become. There were lots of "where do I learn more" questions, for which I need to work on some better answers.