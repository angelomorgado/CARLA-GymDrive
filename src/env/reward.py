'''
Reward Function

This is the file where the reward function can be customized. If you need more information than the provided please also change it in the environment.py file.

I made the reward function based on this data:
- FPS: 30
- Ticks/Steps per second: 100
- Episode time: 30 seconds
- Maximum number of ticks/steps: 3000
'''
from src.carlacore.vehicle import Vehicle
from src.carlacore.world import World
import src.config.configuration as config
import carla
import numpy as np

# ======================================== Global Variables =================================================================
class Reward:
    def __init__(self) -> None:
        self.terminated       = False
        self.inside_stop_area = False
        self.has_stopped      = False
        self.current_steering = 0.0
        self.current_throttle = 0.0
        self.waypoints        = []      
        self.total_ep_reward  = 0  
        
        self.countint = 0

    # ======================================== Main Reward Function ==========================================================
    def calculate_reward(self, vehicle: Vehicle, current_pos, target_pos, next_waypoint_pos, speed) -> float:   
        target_distance = self.distance(current_pos, target_pos)
        next_waypoint_distance = self.distance(current_pos, next_waypoint_pos)
        
        if self.terminated:
            self.countint += 1
            print("The episode already ended!!!, count: ", self.countint)
            
        reward = self.__collision_reward(vehicle) + \
            self.__steering_jerk(vehicle) + \
            self.__throttle_brake_jerk(vehicle) + \
            self.__speed_reward(speed) + \
            self.__target_destination(target_distance) + \
            self.__waypoint_reached(next_waypoint_distance)
        
        self.total_ep_reward += reward
        
        return reward
        
    # ============================================= Reward Functions ==========================================================
    def __collision_reward(self, vehicle):
        '''
        This reward function penalizes the vehicle if it collides with anything or if it leaves its lane. The reward is calculated as follows:
        {
            0.0     : if no collision/lane invasion occurred,
            -lambda : if collision/lane invasion occurred
        }
        
        Based on the calculations, the max reward for this function is 0 and the min reward is -10;
        lambda = 20
        '''
        lbd = 20
        if vehicle.collision_occurred() or vehicle.lane_invasion_occurred():
            self.terminated = True
            return -lbd
        else:
            return 0
        
    def __steering_jerk(self, vehicle, threshold=0.2):
        '''
        This reward function aims to minimize the sudden changes in the steering value of the vehicle. The reward is calculated as follows:
        {
            0.0     : if the steering value difference is less than the threshold,
            -lambda : if the steering value difference is greater or equal than the threshold
        }
        
        Based on the calculations, the max reward for this function is 0 and the min reward is -10;
        lambda = 1/300
        '''
        lbd = 10/config.ENV_MAX_STEPS
        steering_diff = abs(vehicle.get_steering() - self.current_steering)
        self.current_steering = vehicle.get_steering()
        return -lbd if steering_diff > threshold else 0.0

    def __throttle_brake_jerk(self, vehicle, threshold=0.1):
        '''
        This reward function aims to minimize the sudden changes in the throttle/brake of the vehicle. The reward is calculated as follows:
        {
            0.0     : if the throttle/brake difference is less than the threshold,
            -lambda : if the throttle/brake difference is greater or equal than the threshold
        }
        
        Based on the calculations, the max reward for this function is 0 and the min reward is -10;
        lambda = 1/300
        '''
        lbd = 10/config.ENV_MAX_STEPS
        throttle_diff = abs(vehicle.get_throttle_brake() - self.current_throttle)
        self.current_throttle = vehicle.get_throttle_brake()
        return -lbd if throttle_diff > threshold else 0.0

    def __speed_reward(self, speed, speed_limit=50):
        '''
        This reward function is based on the speed of the vehicle. It aims to keep the vehicle at a good speed while preventing it from going over the speed limit. The reward is calculated as follows:
        {
            0       : if speed < 2,
            lambda  : if speed >= 2
            -lambda : if speed > speed_limit
        }
        
        Based on precise calculations the max reward for this function is 15 and the min reward is -15.
        lambda = 15/config.ENV_MAX_STEPS
        '''
        lbd = 15/config.ENV_MAX_STEPS
        
        if speed < 2:
            return 0.0
        elif speed >= 2 and speed <= speed_limit:
            return lbd
        else:
            return -lbd

    def __target_destination(self, target_distance, threshold=5.0):
        '''
        This function rewards the vehicle more generously the closer it gets to the target, and, if it reaches the target, it gives an incredibly high reward, as to tell him that it arrived. The reward is calculated as follows:
        {
            100                                                : if distance <= threshold,
            (-7 * distance + 395) / (9 * config.ENV_MAX_STEPS) : if 5 < distance <= 50,   # More accentuated reward for being closer to the target
            (100 - distance) / (10 * config.ENV_MAX_STEPS)     : if 50 < distance <= 100, # Less accentuated reward for being further from the target
            0                                                  : if distance > 100
        }
        
        Based on precise calculations the max reward for this function is 100 and the min reward is 0.
        '''
        if target_distance <= threshold:
            self.terminated = True
            return 100.0
        elif target_distance > threshold and target_distance <= 50.0:
            return (-7.0*target_distance + 395.0) / (9.0 * config.ENV_MAX_STEPS)
        elif target_distance > 50.0 and target_distance <= 100.0:
            return (100.0 - target_distance) / (10.0 * config.ENV_MAX_STEPS)
        else:
            return 0.0
        
    def __waypoint_reached(self, next_waypoint_distance, threshold=1.0):
        '''
        This reward function gives the agent points if it reaches a waypoint. The reward is calculated as follows:
        {
            2  : if distance < threshold,
            0  : if distance >= threshold
        }
        
        Based on precise calculations the max reward for this function is (2 * n_waypoints) and the min reward is 0.
        
        After the waypoint is reached, it is deleted from the waypoint list (it is the first element). 
        '''
        if next_waypoint_distance < threshold:
            self.waypoints.pop(0)
            return 2.0
        else:
            return 0.0
        
    def __light_pole_trangression(self, map, vehicle, world):
        '''
        This reward function penalizes the agent if it doesn't stop at a stop sign. The reward is calculated as follows:
        {
            0       : if the vehicle stops at the stop sign,
            -lambda : if the vehicle doesn't stop at the stop sign
        }
        
        Based on precise calculations the max reward for this function is 0 and the min reward is -20.
        '''
        lbd = 20.0
        
        # Get the current waypoint of the vehicle
        current_waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True)

        # Get the traffic lights affecting the current waypoint
        traffic_lights = world.get_world().get_traffic_lights_from_waypoint(current_waypoint, distance=10.0)

        for traffic_light in traffic_lights:
            # Check if the traffic light is red
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                # Get the stop waypoints for the traffic light
                stop_waypoints = traffic_light.get_stop_waypoints()

                # Check if the vehicle has passed the stop line
                for stop_waypoint in stop_waypoints:
                    if current_waypoint.transform.location.distance(stop_waypoint.transform.location) < 2.0 and vehicle.get_speed() > 0.3:
                        self.terminated = True
                        return -lbd

        return 0.0

    def __stop_sign_transgression(self, vehicle, map):
        '''
        This reward function penalizes the agent if it doesn't stop at a stop sign. The reward is calculated as follows:
        {
            0       : if the vehicle stops at the stop sign,
            -lambda : if the vehicle doesn't stop at the stop sign
        }
        
        Based on precise calculations the max reward for this function is 0 and the min reward is -20.
        '''
        lbd = 20.0
        distance = 20.0  # meters (adjust as needed)
        
        current_location = vehicle.get_location()
        current_waypoint = map.get_waypoint(current_location, project_to_road=True)
        
        # Get all the stop sign landmarks within a certain distance from the vehicle and on the same road
        stop_signs_on_same_road = []
        for landmark in current_waypoint.get_landmarks_of_type(distance, carla.LandmarkType.StopSign):
            landmark_waypoint = map.get_waypoint(landmark.transform.location, project_to_road=True)
            if landmark_waypoint.road_id == current_waypoint.road_id:
                stop_signs_on_same_road.append(landmark)

        if len(stop_signs_on_same_road) == 0:
            if self.inside_stop_area and self.has_stopped:
                print("Vehicle has stopped at the stop sign.")
                self.has_stopped = False
                self.inside_stop_area = False
                return 0
            elif self.inside_stop_area and not self.has_stopped:
                print("Vehicle has not stopped at the stop sign.")
                self.has_stopped = False
                self.inside_stop_area = False
                self.terminated = True
                return -lbd
            else:            
                return 0.0
        else:
            self.inside_stop_area = True

        # The vehicle entered the stop sign area
        for stop_sign in stop_signs_on_same_road:
            # Check if the vehicle has stopped
            if vehicle.get_speed() < 1.0:
                self.has_stopped = True
        
    # ==================================== Helper Functions ================================================================
    # Distance function between two lists of 3 points
    def distance(self, a, b):
        return np.linalg.norm(a - b)

    def get_waypoints(self):
        return self.waypoints
    
    def reset(self, waypoints):
        self.terminated       = False
        self.inside_stop_area = False
        self.has_stopped      = False
        self.current_steering = 0.0
        self.current_throttle = 0.0
        self.waypoints        = waypoints
        self.total_ep_reward  = 0
    
    def get_terminated(self):
        return self.terminated
    
    def get_total_ep_reward(self):
        return self.total_ep_reward
