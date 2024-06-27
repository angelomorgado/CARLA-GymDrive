'''
MapControl:
    - Module that controls the current map of the simulation, and allows its customization
'''
import carla
import time

class MapControl:
    def __init__(self, world, client):
        self.__world          = world
        self.__client         = client
        self.__available_maps = [m for m in self.__client.get_available_maps() if 'Opt' not in m] # Took out the layered maps
        self.__map_dict       = {m.split("/")[-1]: idx for idx, m in enumerate(self.__available_maps)}
        self.__active_map     = list(self.__map_dict).index(self.__world.get_map().name.split("/")[-1].split("_")[0])
        self.__map            = self.__world.get_map()

    def get_active_map_name(self):
        return self.__map.name.split("/")[-1].split("_")[0]

    def get_map(self):
        return self.__map
    
    def print_available_maps(self):
        for idx, m in enumerate(self.__available_maps):
            print(f'{idx}: {m}')
    
    def set_active_map(self, map_name, reload_map=False):
        # Check if the map is already loaded
        if self.__map_dict[map_name] == self.__active_map and not reload_map:
            return
        
        self.__active_map = self.__map_dict[map_name]
        if map_name in ["Town15", "Town11", "Town12", "Town13"]:
            map_name += f"/{map_name}"
        self.__client.load_world('/Game/Carla/Maps/' + map_name)
        time.sleep(3)
        self.__map = self.__world.get_map()

    # Serves for debugging purposes
    def change_map(self):
        self.print_available_maps()
        map_idx = int(input('Choose a map index: '))
        self.set_active_map(map_idx)
    
    def reload_map(self):
        self.set_active_map(self.get_active_map_name(), reload_map=True)