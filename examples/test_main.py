'''
List of simple examples that can be used to test the different functionalities of the template.
'''

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vehicle import Vehicle
from src.world import World
from src.server import CarlaServer

# Change the vehicle's physics to a determined weather that is stated in the JSON file specified in the config file.
def physics_main():
    # Carla client
    world = World()
    
    # Create vehicle
    autonomous_vehicle = Vehicle(world=world)
    autonomous_vehicle.set_autopilot(True)
    autonomous_vehicle.print_vehicle_physics()

    # Change the vehicle's physics to a determined weather that is stated in the JSON file.
    autonomous_vehicle.change_vehicle_physics("wet")

    print("\n\n===========================================================================================================\n")
    autonomous_vehicle.print_vehicle_physics()

    autonomous_vehicle.destroy_vehicle()

# Control the vehicle
def control_main():
    # Carla client
    world = World()
    world.set_active_map_name('/Game/Carla/Maps/Town01')
    
    # Create vehicle
    autonomous_vehicle = Vehicle(world=world.get_world())
    autonomous_vehicle.spawn_vehicle() # Spawn vehicle at random location

    world.place_spectator_above_location(autonomous_vehicle.get_location())

    # [Steer (-1.0, 1.0), Throttle (0.0, 1.0), Brake (0.0, 1.0)]
    action = [0.0, 1.0, 0.0]
    discrete_action = 0

    while True:
        try:
            # Continuous action
            autonomous_vehicle.control_vehicle(action)
            # Discrete action
            # autonomous_vehicle.control_vehicle_discrete(discrete_action)
        except KeyboardInterrupt:
            autonomous_vehicle.destroy_vehicle()
            CarlaServer.close_server()
            break

# Spawn traffic
def traffic_main():
    world = World()

    world.spawn_vehicles(20, autopilot_on=True)
    world.spawn_pedestrians(20)
    while True:
        try:
            print("", end="")
            pass
        except KeyboardInterrupt:
            print("Exiting...")
            world.destroy_vehicles()
            world.destroy_pedestrians()
            break

# Change the weather
def weather_main():
    world = World()

    while True:
        try:
            world.choose_weather()
        except KeyboardInterrupt:
            break
        
# Change the map
def map_main():
    world = World()

    while True:
        try:
            world.change_map()
        except KeyboardInterrupt:
            break

# Open and close the server
def server_main():
    process = CarlaServer.initialize_server(low_quality=True)
    CarlaServer.close_server(process)

if __name__ == '__main__':
    CarlaServer.initialize_server(low_quality=True)
    control_main()
    CarlaServer.close_server()
