import carla
from pynput import keyboard

'''
Kayboard Control Module:
    It provides the functionality to control a vehicle using the keyboard.
    
    Keys:
        - w: throttle
        - s: brake
        - a: steer left
        - d: steer right
        - q: toggle reverse
        - z: toggle lock (toggles vehicle control on/off)
'''
class KeyboardControl:
    def __init__(self, vehicle):
        self.__vehicle = vehicle
        self.__throttle = 0.0
        self.__brake = 0.0
        self.__steering = 0.0
        self.__reverse = False
        self.__lock = False

        # Create a listener that will call on_press and on_release when a key is pressed or released
        self.__listener = keyboard.Listener(on_press=self.__on_press, on_release=self.__on_release)
        self.__listener.start()

    def __on_press(self, key):
        try:
            if key.char == 'w':
                self.__throttle = 1.0
            elif key.char == 's':
                self.__brake = 1.0
            elif key.char == 'a':
                self.__steering = -1.0
            elif key.char == 'd':
                self.__steering = 1.0
            elif key.char == 'q':
                self.__reverse = not self.__reverse
        except AttributeError:
            pass

    def __on_release(self, key):
        try:
            if key.char == 'w':
                self.__throttle = 0.0
            elif key.char == 's':
                self.__brake = 0.0
            elif key.char in ('a', 'd'):
                self.__steering = 0.0
            elif key.char == 'z':
                self.__lock = not self.__lock
        except AttributeError:
            pass

    def apply_controls(self):
        if not self.__lock:
            # Apply controls to the vehicle
            control = carla.VehicleControl()
            if self.__reverse:
                control.reverse = True
            control.throttle = self.__throttle
            control.brake = self.__brake
            control.steer = self.__steering
            self.__vehicle.apply_control(control)

    def tick(self):
        # Apply controls to the vehicle
        self.apply_controls()
    
    def clean(self):
        self.__listener.stop()
        self.__listener.join()
