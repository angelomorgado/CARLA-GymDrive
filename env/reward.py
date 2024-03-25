from src.vehicle import Vehicle
from src.world import World
import configuration as config
import carla
import numpy as np

terminated = False
inside_stop_area = False
has_stopped = False

# ======================================== Main Reward Function ==========================================================
# If you change this function's signature, you must change the signature of the function in the environment.py file!!
def calculate_reward(vehicle: Vehicle, world: World, map: carla.Map, scenario_dict, num_steps: int, time_limit_reached: bool) -> float:
    global terminated
    vehicle_location = vehicle.get_location()
    waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    reward_lambdas = config.ENV_REWARDS_LAMBDAS
    terminated = False
    
    return reward_lambdas['orientation'] * __get_orientation_reward(waypoint, vehicle) + \
           reward_lambdas['distance'] * __get_distance_reward(waypoint, vehicle_location) + \
           reward_lambdas['speed'] * __get_speed_reward(vehicle) + \
           reward_lambdas['destination'] * __get_destination_reward(vehicle_location, scenario_dict, num_steps) + \
           reward_lambdas['collision'] * __get_collision_reward(vehicle) + \
           reward_lambdas['light_pole_transgression'] * __get_light_pole_trangression_reward(map, vehicle, world) + \
           reward_lambdas['stop_sign_transgression'] * __get_stop_sign_reward(vehicle, map) + \
           reward_lambdas['time_limit'] * __get_time_limit_reward(time_limit_reached) + \
           reward_lambdas['time_driving'] * __get_time_driving_reward(vehicle), terminated

# ============================================= Reward Functions ==========================================================
# This reward is based on the orientation of the vehicle according to the waypoint of where the vehicle is
# R_orientation = \lambda * cos(\theta), where \theta is the angle between the vehicle and the waypoint
def __get_orientation_reward(waypoint, vehicle):
    vh_yaw = __correct_yaw(vehicle.get_vehicle().get_transform().rotation.yaw)
    wp_yaw = __correct_yaw(waypoint.transform.rotation.yaw)

    return np.cos((vh_yaw - wp_yaw)*np.pi/180.)

# This reward is based on the distance between the vehicle and the waypoint
def __get_distance_reward(waypoint, vehicle_location):
    x_wp = waypoint.transform.location.x
    y_wp = waypoint.transform.location.y

    x_vehicle = vehicle_location.x
    y_vehicle = vehicle_location.y

    return np.linalg.norm([x_wp - x_vehicle, y_wp - y_vehicle])

def __get_speed_reward(vehicle, speed_limit=50):
    vehicle_speed = vehicle.get_speed()
    return vehicle_speed - speed_limit if vehicle_speed > speed_limit else 0.0

# This reward is based on if the vehicle reached the destination. the reward will be based on the number of steps taken to reach the destination. The less steps, the higher the reward, but reaching the destination is the highest reward
def __get_destination_reward(current_position, scenario_dict, num_steps, threshold=2.0): 
    global terminated
    current_position = np.array([current_position.x, current_position.y, current_position.z])
    target_position = (scenario_dict['target_position']['x'], scenario_dict['target_position']['y'], scenario_dict['target_position']['z'])
    
    if np.linalg.norm(current_position - target_position) < threshold:
        terminated = True
        return max(num_steps * (1 / config.ENV_MAX_STEPS) + 1, 0.35)
    else:
        return 0

# Collision with other vehicles or pedestrians and even lane invasions
def __get_collision_reward(vehicle):
    global terminated
    if vehicle.collision_occurred() or vehicle.lane_invasion_occurred():
        terminated = True
        return 1
    else:
        return 0

def __get_light_pole_trangression_reward(map, vehicle, world):
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
                if current_waypoint.transform.location.distance(stop_waypoint.transform.location) < 2.0 and vehicle.get_speed() > 0.1:
                    return 1

    return 0

def __get_stop_sign_reward(vehicle, map):
    global inside_stop_area, has_stopped, terminated        
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
        if inside_stop_area and has_stopped:
            print("Vehicle has stopped at the stop sign.")
            has_stopped = False
            inside_stop_area = False
            return 0
        elif inside_stop_area and not has_stopped:
            print("Vehicle has not stopped at the stop sign.")
            has_stopped = False
            inside_stop_area = False
            return 1
        else:            
            return 0
    else:
        inside_stop_area = True

    # The vehicle entered the stop sign area
    for stop_sign in stop_signs_on_same_road:
        # Check if the vehicle has stopped
        if vehicle.get_speed() < 1.0:
            has_stopped = True

# TODO: I think it's not working properly
def __get_time_limit_reward(time_limit_reached):
    return 1 if time_limit_reached else 0

def __get_time_driving_reward(vehicle):
    global terminated
    return 1 if not terminated and vehicle.get_speed() > 1.0 else 0

# ==================================== Helper Functions ================================================================
# This function is used to correct the yaw angle to be between 0 and 360 degrees
def __correct_yaw(x):
    return(((x%360) + 360) % 360)
