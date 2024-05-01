import pygame
import time
from gpiozero import Servo

# Function to clamp PWM values to ensure they are within the valid range
def clamp(min_value, max_value, value):
    return max(min_value, min(max_value, value))

# Initialize pygame and the joystick
pygame.init()
pygame.joystick.init()

# Setup the PWM LED on pin 18
# pwm_led = Servo(18)
min_pulse_width = 0.0006  # 0.6ms
max_pulse_width = 0.0024  # 2.4ms

pwm = Servo(18, min_pulse_width=min_pulse_width, max_pulse_width=max_pulse_width)


try:
    # Try to use the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print("Joystick initialized")
except pygame.error:
    print("No joystick found. Please connect a joystick.")
    pygame.quit()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.JOYAXISMOTION:
            if event.axis == 0:  # We are only interested in axis 0
                # The joystick value ranges from -1 to 1. We need to transform this to 0 to 1.
                pwm_value = event.value
                pwm_value = clamp(-1, 1, pwm_value)  # Ensure PWM value stays within 0 and 1
                pwm.value = pwm_value  # Set PWM value
                print(f"Set PWM value to {pwm_value:.2f} based on joystick axis 0 position")

    # Delay to keep loop manageable
    time.sleep(0.1)

pygame.quit()
