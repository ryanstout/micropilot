import pygame
from gpiozero import Servo
from time import sleep

# Create a Servo object attached to GPIO 18
servo = Servo(18)


# Initialize Pygame
pygame.init()

# Initialize the joystick
pygame.joystick.init()

try:
    # Attempt to setup the joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Joystick initialized")
except pygame.error:
    print("No joystick found.")

# Game loop to keep the window open
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.JOYAXISMOTION:
            # Print the values of the axes
            print("Joystick Axis", event.axis, "value:", event.value)

            if event.axis == 0:
                joystick_val = event.value
                if joystick_val > 1.0:
                    joystick_val = 1.0
                elif joystick_val < -1.0:
                    joystick_val = -1.0

                print("Joystick Val: ", joystick_val)
                servo.value = joystick_val

    # Update loop runs at a manageable pace
    pygame.time.Clock().tick(6)

pygame.quit()
