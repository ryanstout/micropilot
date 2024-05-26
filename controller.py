# Utility file for figuring out the input axis's on the controller

import pygame

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
        elif event.type == pygame.JOYBUTTONDOWN:
            print("Joystick Button", event.button, "pressed.")
        elif event.type == pygame.JOYBUTTONUP:
            print("Joystick Button", event.button, "released.")
    
    # Update loop runs at a manageable pace
    pygame.time.Clock().tick(60)

pygame.quit()
