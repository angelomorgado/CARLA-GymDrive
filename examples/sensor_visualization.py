'''
sensor_visualization.py
- This script is used to visualize the sensors of the vehicle.
'''
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.world import World
from src.display import Display
from src.vehicle import Vehicle
from src.server import CarlaServer

def main():
    CarlaServer.initialize_server()
    world = World()
    
    ego_vehicle = Vehicle(world.get_world())
    ego_vehicle.spawn_vehicle()
    ego_vehicle.set_autopilot(True)
    
    world.place_spectator_above_location(ego_vehicle.get_location())
    
    display = Display('Sensor Visualization', ego_vehicle)
    
    while True:
        try:
            world.tick()
            display.play_window_tick()
        except KeyboardInterrupt:
            display.close_window()
            ego_vehicle.destroy_vehicle()
            world.destroy_world()
            CarlaServer.close_server()

if __name__ == '__main__':
    main()
