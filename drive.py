import pygame
from picarx import Picarx
from time import sleep


px = Picarx()
px.forward(0)

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
            # print("Joystick Axis", event.axis, "value:", event.value)

            #adjusted_value = 

            if event.axis == 3:
                joystick_val = event.value
                if joystick_val > 1.0:
                    joystick_val = 1.0
                elif joystick_val < -1.0:
                    joystick_val = -1.0

                print("Joystick Val: ", joystick_val, " -- axis", event.axis)
                px.set_dir_servo_angle(joystick_val * 25)
            elif event.axis == 1:
                val = event.value
                if val < 0.5 and val > -0.5:
                    val = 0.0
                print("Forward: ", val * -1.0, " -- axis", event.axis)
                px.forward(val * -1.0)

    # Update loop runs at a manageable pace
    pygame.time.Clock().tick(6)

pygame.quit()
