'''
check_infractions.py

- This script is used to check for any infractions by the ego vehicle. The vehicle is controlled by the keyboard.
'''

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vehicle import Vehicle
import configuration
from src.display import Display
from src.world import World
from src.server import CarlaServer
from src.keyboard_control import KeyboardControl
import carla
import random

def check_stop_sign(vehicle, world):
    global has_stopped
    global inside_stop_area
    
    distance = 30.0  # meters (adjust as needed)
    
    current_location = vehicle.get_location()
    current_waypoint = world.get_map().get_waypoint(current_location, project_to_road=True)
    
    # Get all the stop sign landmarks within a certain distance from the vehicle and on the same road
    stop_signs_on_same_road = []
    for landmark in current_waypoint.get_landmarks_of_type(distance, carla.LandmarkType.StopSign):
        landmark_waypoint = world.get_map().get_waypoint(landmark.transform.location, project_to_road=True)
        if landmark_waypoint.road_id == current_waypoint.road_id:
            stop_signs_on_same_road.append(landmark)

    if len(stop_signs_on_same_road) == 0:
        if inside_stop_area and has_stopped:
            print("Vehicle has stopped at the stop sign.")
            has_stopped = False
            inside_stop_area = False
        elif inside_stop_area and not has_stopped:
            print("Vehicle has not stopped at the stop sign.")
            has_stopped = False
            inside_stop_area = False
        else:            
            return
    else:
        inside_stop_area = True

    # The vehicle entered the stop sign area
    for stop_sign in stop_signs_on_same_road:
        # Check if the vehicle has stopped
        if vehicle.get_speed() < 1.0:  # Adjust this threshold for stopped speed
            has_stopped = True

def has_passed_red_light(vehicle, world):
    # Get the current waypoint of the vehicle
    current_waypoint = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True)

    # Get the traffic lights affecting the current waypoint
    traffic_lights = world.get_traffic_lights_from_waypoint(current_waypoint, distance=10.0)

    for traffic_light in traffic_lights:
        # Check if the traffic light is red
        if traffic_light.get_state() == carla.TrafficLightState.Red:
            # Get the stop waypoints for the traffic light
            stop_waypoints = traffic_light.get_stop_waypoints()

            # Check if the vehicle has passed the stop line
            for stop_waypoint in stop_waypoints:
                if current_waypoint.transform.location.distance(stop_waypoint.transform.location) < 2.0:
                    print("Vehicle has passed the red light!!")

        
def main():
    # Carla server
    # server_process = CarlaServer.initialize_server()

    # Carla client
    world = World(synchronous_mode=True)
    world.set_active_map('Town07')
    # world.set_random_weather()

    # Create vehicle
    autonomous_vehicle = Vehicle(world=world.get_world())
    autonomous_vehicle.spawn_vehicle()  # Spawn vehicle at random location
    v_control = KeyboardControl(autonomous_vehicle.get_vehicle()) # Control vehicle with keyboard
    
    
    world.place_spectator_above_location(autonomous_vehicle.get_location())

    # Create display
    display = Display('Carla Sensor feed', autonomous_vehicle)
    
    # Traffic and pedestrians
    # world.spawn_vehicles_around_ego(autonomous_vehicle.get_vehicle(), num_vehicles_around_ego=40, radius=150)
    # world.spawn_pedestrians_around_ego(autonomous_vehicle.get_location(), num_pedestrians=40, radius=150)  

    while True:
        try:
            world.tick()
            v_control.tick()
            display.play_window_tick()
        except KeyboardInterrupt:
            autonomous_vehicle.destroy_vehicle()
            display.close_window()
            # CarlaServer.kill_carla_linux()
            break

if __name__ == '__main__':
    main()