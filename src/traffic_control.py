import carla

import random
import time
import numpy as np
import math

import configuration as config

'''
Traffic Controller module:
    It provides the functionality to spawn, destroy, and control vehicles and pedestrians in the Carla simulation.
'''

class TrafficControl:
    def __init__(self, world) -> None:
        self.__active_vehicles = []
        self.__active_pedestrians = []
        self.__active_ai_controllers = []
        self.__world = world
        self.__map = None
        
    def update_map(self, map):
        self.__map = map

    # ============ Vehicle Control ============
    def spawn_vehicles(self, num_vehicles = 10, autopilot_on = False):
        if num_vehicles < 1:
            print("You need to spawn at least 1 vehicle.")
            return
        
        if config.VERBOSE:
            print(f"Spawning {num_vehicles} vehicle(s)...")
        
        vehicle_bp = self.__world.get_blueprint_library().filter('vehicle.*')
        spawn_points = self.__map.get_spawn_points()

        for i in range(num_vehicles):
            vehicle = None
            while vehicle is None:
                spawn_point = random.choice(spawn_points)
                transform = carla.Transform(
                    spawn_point.location,
                    spawn_point.rotation
                )
                try:
                    vehicle = self.__world.try_spawn_actor(random.choice(vehicle_bp), transform)
                except:
                    # try again if failed to spawn vehicle
                    continue
            
            self.__active_vehicles.append(vehicle)
            # time.sleep(0.1)
        if config.VERBOSE:
            print('Successfully spawned {} vehicles!'.format(num_vehicles))
    
    def destroy_vehicles(self):
        for vehicle in self.__active_vehicles:
            vehicle.set_autopilot(False)
            try:
                vehicle.destroy()
            except RuntimeError as e:
                continue
        self.__active_vehicles = []
        if config.VERBOSE:
            print('Destroyed all vehicles!')
    
    def toggle_autopilot(self, autopilot_on = True):
        for vehicle in self.__active_vehicles:
            vehicle.set_autopilot(autopilot_on)

    def spawn_vehicles_around_ego(self, ego_vehicle, radius, num_vehicles_around_ego, seed=None):
        if seed is not None:
            random.seed(seed)

        self.spawn_points = self.__map.get_spawn_points()
        ego_location = ego_vehicle.get_location()
        accessible_points = []

        for spawn_point in self.spawn_points:
            dis = math.sqrt((ego_location.x - spawn_point.location.x)**2 +
                            (ego_location.y - spawn_point.location.y)**2)
            # it also can include z-coordinate, but it is unnecessary
            if dis < radius and dis > 5.0:
                accessible_points.append(spawn_point)

        vehicle_bps = self.__world.get_blueprint_library().filter('vehicle.*.*') 

        if len(accessible_points) < num_vehicles_around_ego:
            num_vehicles_around_ego = len(accessible_points)

        for i in range(num_vehicles_around_ego):
            point = accessible_points[i]
            vehicle_bp = random.choice(vehicle_bps)
            try:
                vehicle = self.__world.spawn_actor(vehicle_bp, point)
                vehicle.set_autopilot(True)
                self.__active_vehicles.append(vehicle)
            except:
                print('Error: Failed to spawn a traffic vehicle.')
                pass

    def toggle_lights(self, lights_on=True):
        for vehicle in self.__active_vehicles:
            if lights_on:
                vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
            else:
                vehicle.set_light_state(carla.VehicleLightState.NONE)
            
    # ============ Pedestrian Control ============
    def spawn_pedestrians(self, num_walkers=10):
        if num_walkers < 1:
            print("You need to spawn at least 1 pedestrian.")
            return
        
        walker_controller_bp = self.__world.get_blueprint_library().find('controller.ai.walker')
        walker_bps = self.__world.get_blueprint_library().filter('walker.pedestrian.*')

        # Get spawn points on sidewalks
        spawn_points = self.__map.get_spawn_points()

        for _ in range(num_walkers):
            # Randomly select a spawn point
            spawn_point = random.choice(spawn_points)
            
            # Extract location from the spawn point
            spawn_location = spawn_point.location

            # Get waypoint from the location
            sidewalk_waypoint = self.__map.get_waypoint(spawn_location, project_to_road=True, lane_type=(carla.LaneType.Sidewalk))

            # Spawn walker and controller at the sidewalk waypoint.
            walker_bp = random.choice(walker_bps)
            try:
                walker = self.__world.spawn_actor(walker_bp, carla.Transform(sidewalk_waypoint.transform.location))
            except RuntimeError:
                continue
            self.__active_pedestrians.append(walker)

            walker_controller = self.__world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
            self.__active_ai_controllers.append(walker_controller)
            
            # Keep the commented code if you want to start and move the walkers
            # walker_controller.start()
            # walker_controller.go_to_location(self.__world.get_random_location_from_navigation())

        if config.VERBOSE:
            print("Spawned", num_walkers, "walkers on random sidewalks.")
    
    def spawn_pedestrians_around_ego(self, vehicle_location, num_walkers=10, radius=25.0):
        if num_walkers < 1:
            print("You need to spawn at least 1 pedestrian.")
            return
        
        walker_controller_bp = self.__world.get_blueprint_library().find('controller.ai.walker')
        
        for _ in range(num_walkers):

            # Find a sidewalk waypoint within a radius of the vehicle location.
            sidewalk_waypoint = None
            while sidewalk_waypoint is None:
                random_offset = carla.Location(
                    x=random.uniform(-radius, radius),
                    y=random.uniform(-radius, radius))
                potential_location = vehicle_location + random_offset
                waypoint = self.__map.get_waypoint(potential_location, project_to_road=True, lane_type=(carla.LaneType.Sidewalk))
                if waypoint:
                    sidewalk_waypoint = waypoint

            # Spawn walker and controller at the sidewalk waypoint.
            walker_bp = random.choice(self.__world.get_blueprint_library().filter('walker.pedestrian.*'))
            try:
                walker = self.__world.spawn_actor(walker_bp, sidewalk_waypoint.transform)
            except RuntimeError:
                continue
            self.__active_pedestrians.append(walker)

            walker_controller = self.__world.spawn_actor(walker_controller_bp, carla.Transform(), walker)
            self.__active_ai_controllers.append(walker_controller)
            
            # Gives off segmenation fault: Carla's fault!! I did according to the documentation!!
            # walker_controller.start()
            # walker_controller.go_to_location(self.__world.get_random_location_from_navigation())

        if config.VERBOSE:
            print("Spawned", num_walkers, "walkers near the vehicle.")

    def destroy_pedestrians(self):
        for idx, pedestrian in enumerate(self.__active_pedestrians):
            try:
                pedestrian.destroy()
                self.__active_ai_controllers[idx].stop()
                self.__active_ai_controllers[idx].destroy()
            except Exception as e:
                print(f"Error destroying pedestrians: {e}")

        self.__active_pedestrians = []
        self.__active_ai_controllers = []
        if config.VERBOSE:
            print('Destroyed all pedestrians!')
