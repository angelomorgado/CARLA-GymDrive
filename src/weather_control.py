import carla

import re
import random

'''
This module provides functions to control weather in the simulator

Currently available weather presets:

'Clear Night'
'Clear Noon'
'Clear Sunset'
'Cloudy Night'
'Cloudy Noon'
'Cloudy Sunset'
'Default'
'Dust Storm'
'Hard Rain Night'
'Hard Rain Noon'
'Hard Rain Sunset'
'Mid Rain Sunset'
'Mid Rainy Night'
'Mid Rainy Noon'
'Soft Rain Night'
'Soft Rain Noon'
'Soft Rain Sunset'
'Wet Cloudy Night'
'Wet Cloudy Noon'
'Wet Cloudy Sunset'
'Wet Night'
'Wet Noon'
'Wet Sunset'
'''

class WeatherControl:
    def __init__(self, world):
        self.__weather_list = self.__get_all_weather_presets()
        self.__active_weather = "Default"
        self.__world = world

    def __get_all_weather_presets(self):
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
    
    # The output is a tuple (carla.WeatherPreset, Str: name of the weather preset)
    def get_weather_presets(self):
        return self.__weather_list
    
    def get_active_weather(self):
        return self.__active_weather
    
    def print_all_weather_presets(self):    
        for idx, weather in enumerate(self.__weather_list):
            print(f'{idx}: {weather[1]}')

    def __activate_weather_preset(self, idx):
        self.__world.set_weather(self.__weather_list[idx][0])

    def set_active_weather_preset(self, weather):
        for idx, w in enumerate(self.__weather_list):
            if w[1] == weather:
                self.__active_weather = w[1]
                self.__activate_weather_preset(idx)
                return
    
    def set_random_weather_preset(self):
        idx = random.randint(0, len(self.__weather_list) - 1)
        self.__active_weather = self.__weather_list[idx][1]
        self.__activate_weather_preset(idx)

    # This method let's the user choose with numbers the active preset. It serves as more of a debug.
    def choose_weather(self):
        print('Choose a weather preset:')
        for idx, weather in enumerate(self.__weather_list):
            print(f'{idx}: {weather[1]}')
        
        idx = int(input())

        try:
            self.__active_weather = self.__weather_list[idx][1]
            self.__activate_weather_preset()
        except IndexError:
            print('Invalid index')
            return
        
        print(f'Weather preset {self.__active_weather} activated')
