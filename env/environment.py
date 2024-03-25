'''
Environment class for the carla environment.

It is a wrapper around the carla environment, and it is used to interact with the environment in a more convenient way.

It implements the following methods:
 - reset: resets the environment and returns the initial state
 - step: takes an action and returns the next state, the reward, a flag indicating if the episode is done, and a dictionary with extra information
 - close: closes the environment

Observation Space:
    [RGB image, LiDAR point cloud, Current position, Target position, Current situation]

    The current situation cannot be a string therefore it was converted to a numerical value using a dictionary to map the string to a number

    Dict:{
        Road:       0,
        Roundabout: 1,
        Junction:   2,
        Tunnel:     3,
    }

Action Space:
    Continuous:from gymnasium import spaces
        [Steering (-1.0, 1.0), Throttle/Brake (-1.0, 1.0)]
    Discrete:
        [Action] (0: Accelerate, 1: Decelerate, 2: Left, 3: Right) <- It's a number from 0 to 3

'''

import numpy as np
import json
import time
import random
import carla

import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id="carla-rl-gym-v0", # name-version
    entry_point="env.environment:CarlaEnv",
    max_episode_steps=10000,
)

from src.world import World
from src.server import CarlaServer
from src.vehicle import Vehicle
from src.display import Display
import configuration as config
from env.reward import calculate_reward
import env.observation_action_space
from env.pre_processing import PreProcessing

# Name: 'carla-rl-gym-v0'
class CarlaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": config.SIM_FPS}
    def __init__(self, continuous=True, scenarios=[], time_limit=60, initialize_server=True, random_weather=False, random_traffic=False, synchronous_mode=True, show_sensor_data=False, has_traffic=True, verbose=True):
        super().__init__()
        # Read the environment settings
        self.__is_continuous = continuous
        self.__random_weather = random_weather
        self.__random_traffic = random_traffic
        self.__synchronous_mode = synchronous_mode
        self.__show_sensor_data = show_sensor_data
        self.__has_traffic = has_traffic
        self.__verbose = verbose
        self.__automatic_server_initialization = initialize_server

        # 1. Start the server
        if self.__automatic_server_initialization:
            self.__server_process = CarlaServer.initialize_server(low_quality = config.SIM_LOW_QUALITY, offscreen_rendering = config.SIM_OFFSCREEN_RENDERING)
        
        # 2. Connect to the server
        self.__world = World(synchronous_mode=self.__synchronous_mode)

        # 3. Read the flag and get the appropriate situations
        self.__get_situations(scenarios)
        # 4. Create the vehicle
        self.__vehicle = Vehicle(self.__world.get_world())

        # 5. Observation space:
        self.observation_space = env.observation_action_space.observation_space
        self.__observation = None
        self.pre_processing = PreProcessing()

        # 6: Action space
        if self.__is_continuous:
            # For continuous actions
            self.action_space = env.observation_action_space.continuous_action_space
        else:
            # For discrete actions
            self.action_space = env.observation_action_space.discrete_action_space
        
        # Truncated flag
        self.__time_limit = time_limit
        self.__time_limit_reached = False
        self.__truncated = False  # Used for an episode that was terminated due to a time limit or errors

        # Variables to store the current state
        self.__active_scenario_name = None
        self.__active_scenario_dict = None

        self.__situations_map = env.observation_action_space.situations_map

        # Auxiliar variables
        self.__first_episode = True
        
    # ===================================================== GYM METHODS =====================================================                
    # This reset loads a random scenario and returns the initial state plus information about the scenario
    # Options may include the name of the scenario to load    
    def reset(self, seed=None, options={'scenario_name': None}):
        # 1. Choose a scenario
        if options['scenario_name'] is not None:
            self.__active_scenario_name = options['scenario_name']
        else:
            self.__active_scenario_name = self.__chose_situation(seed)
        
        # 2. Load the scenario
        print(f"Loading scenario {self.__active_scenario_name}...")
        try:
            self.load_scenario(self.__active_scenario_name, seed)
        except KeyboardInterrupt as e:
            self.clean_scenario()
            print("Scenario loading interrupted!")
            exit(0)
        print("Scenario loaded!")
        
        # 3. Place the spectator
        self.place_spectator_above_vehicle()
        
        # 4. Get the initial state (Get the observation data)
        time.sleep(0.5)
        self.__update_observation()
        
        # 5. Start the timer
        self.__start_timer()
        print("Episode started!")
        
        self.number_of_steps = 0
        # Return the observation and the scenario information
        return self.__observation, self.__active_scenario_dict
    
    def render(self, mode='human'):
        if mode == 'human':
            self.__world.tick()
            self.display.play_window_tick()
        else:
            raise NotImplementedError("This mode is not implemented yet")

    def step(self, action):
        # 0. Tick the world if in synchronous mode
        if self.__synchronous_mode:
            try:
                self.__world.tick()
            except KeyboardInterrupt:
                self.clean_scenario()
                print("Episode interrupted!")
                exit(0)
        self.number_of_steps += 1
        # 1. Control the vehicle
        self.__control_vehicle(np.array(action))
        # 1.5 Tick the display if it is active
        if self.__show_sensor_data:
            self.display.play_window_tick()
        # 2. Update the observation
        self.__update_observation()
        # 3. Calculate the reward
        reward, terminated = calculate_reward(self.__vehicle, self.__world, self.__map, self.__active_scenario_dict, self.number_of_steps, self.__time_limit_reached)
        # 5. Check if the episode is truncated
        try:
            self.__truncated = self.__timer_truncated()
        except KeyboardInterrupt:
            self.clean_scenario()
            print("Episode interrupted!")
            exit(0)
        if self.__truncated or terminated:
            self.clean_scenario()
        # 5. Return the observation, the reward, the terminated flag and the scenario information
        return self.__observation, reward, terminated, self.__truncated, self.__active_scenario_dict

    # Closes everything, more precisely, destroys the vehicle, along with its sensors, destroys every npc and then destroys the world
    def close(self):
        # 1. Destroy the vehicle
        self.__vehicle.destroy_vehicle()
        # 2. Destroy pedestrians and traffic vehicles
        self.__world.destroy_vehicles()
        self.__world.destroy_pedestrians()
        # 2. Destroy the world
        self.__world.destroy_world()
        # 3. Close the server
        if self.__automatic_server_initialization:
            CarlaServer.close_server(self.__server_process)


    # ===================================================== OBSERVATION/ACTION METHODS =====================================================
    def __update_observation(self):        
        observation_space = self.__vehicle.get_observation_data()
        rgb_image = observation_space[0]
        lidar_point_cloud = observation_space[1]
        current_position = observation_space[2]
        target_position = np.array([self.__active_scenario_dict['target_gnss']['lat'], self.__active_scenario_dict['target_gnss']['lon'], self.__active_scenario_dict['target_gnss']['alt']])
        situation = self.__situations_map[self.__active_scenario_dict['situation']]

        observation = {
            'rgb_data': np.uint8(rgb_image),
            'lidar_data': np.float32(lidar_point_cloud),
            'position': np.float32(current_position),
            'target_position': np.float32(target_position),
            'situation': situation
        }
        
        self.__observation = self.pre_processing.preprocess_data(observation)

    # ===================================================== SCENARIO METHODS =====================================================
    def load_scenario(self, scenario_name, seed=None):
        try:
            scenario_dict = self.situations_dict[scenario_name]
        except KeyError:
            new_name = self.__choose_random_situation(seed)
            scenario_dict = self.situations_dict[new_name]
            print(f"Scenario {scenario_name} not found! Loading random scenario {new_name}...")
        self.__active_scenario_name = scenario_name
        self.__seed = seed
        self.__active_scenario_dict = scenario_dict
         
        # World
        # This is a fix to a weird bug that happens when the first town is the same as the default map (comment and run a couple of times to see the bug)
        if self.__first_episode and self.__active_scenario_dict['map_name'] == self.__world.get_active_map_name():
            self.__world.reload_map()
        self.__first_episode = False
        
        self.__load_world(scenario_dict['map_name'])
        self.__map = self.__world.update_traffic_map()
        time.sleep(2.0)
        if self.__verbose:
            print("World loaded!")
        
        # Weather
        self.__load_weather(scenario_dict['weather_condition'])
        if self.__verbose:
            print(self.__world.get_active_weather(), " weather preset loaded!")
        
        # Ego vehicle
        self.__spawn_vehicle(scenario_dict)
        if self.__show_sensor_data:   
            self.display = Display('Ego Vehicle Sensor feed', self.__vehicle)
            self.display.play_window_tick()
        if self.__verbose:
            print("Vehicle spawned!")
        
        # Traffic
        if self.__has_traffic:
            self.__spawn_traffic(seed=seed)
            # self.__world.spawn_pedestrians_around_ego(self.__vehicle.get_location(), num_pedestrians=10)
            if self.__verbose:
                print("Traffic spawned!")
        self.__toggle_lights()

    def clean_scenario(self):
        self.__vehicle.destroy_vehicle()
        self.__world.destroy_vehicles()
        self.__world.destroy_pedestrians()
        if self.__verbose:
            print("Scenario cleaned!")
    
    def print_all_scenarios(self):
        for idx, i in enumerate(self.situations_list):
            print(idx, ": ", i)
    
    def __load_world(self, name):
        self.__world.set_active_map(name)
        
    def __spawn_vehicle(self, s_dict):
        location = (s_dict['initial_position']['x'], s_dict['initial_position']['y'], s_dict['initial_position']['z'])
        rotation = (s_dict['initial_rotation']['pitch'], s_dict['initial_rotation']['yaw'], s_dict['initial_rotation']['roll'])
        self.__vehicle.spawn_vehicle(location, rotation)
    
    def __toggle_lights(self):
        if "night" in self.__world.get_active_weather().lower() or "noon" in self.__world.get_active_weather().lower():
            self.__world.toggle_lights(lights_on=True)
            self.__vehicle.toggle_lights(lights_on=True)
        else:
            self.__world.toggle_lights(lights_on=False)
            self.__vehicle.toggle_lights(lights_on=False)

    def __load_weather(self, weather_name):
        if self.__random_weather:
            self.__world.set_random_weather()
        else:
            self.__world.set_active_weather_preset(weather_name)
    
    # If the seed is not none send the seed, else make the scenario based on its name
    def __spawn_traffic(self, seed):
        if not self.__random_traffic and self.__active_scenario_dict['traffic_density'] == 'None':
            return

        # The traffic isn't random, so it will be based on the scenario name
        if not self.__random_traffic:
            random.seed(self.__active_scenario_name)
            seed = self.__active_scenario_name
        
        if seed is not None:
            random.seed(seed)
        
        # Give density to the traffic
        if not self.__random_traffic:
            if self.__active_scenario_dict['traffic_density'] == 'Low':
                num_vehicles = random.randint(1, 5)
            else:
                num_vehicles = random.randint(10, 20)
        else:
            num_vehicles = random.randint(1, 20)
        
        self.__world.spawn_vehicles_around_ego(self.__vehicle.get_vehicle(), radius=100, num_vehicles_around_ego=num_vehicles, seed=seed)
    
    def __choose_random_situation(self, seed=None):
        if seed:
            np.random.seed(seed)
        return np.random.choice(self.situations_list)

    def __chose_situation(self, seed):
        if isinstance(seed, str):
            print("Seed needs to be an integer! Loading a random scenario...")
            return self.__choose_random_situation()
        else:
            return self.__choose_random_situation(seed)
    
    # ===================================================== SITUATIONS PARSING =====================================================
    # Filter the current situations based on the flag
    def __get_situations(self, scenarios):
        with open(config.ENV_SCENARIOS_FILE, 'r') as f:
            self.situations_dict = json.load(f)

        if scenarios:
            self.situations_dict = {key: value for key, value in self.situations_dict.items() if value['situation'] in scenarios}

        self.situations_list = list(self.situations_dict.keys())

            
    # ===================================================== AUX METHODS =====================================================
    def __control_vehicle(self, action):
        if self.__is_continuous:
            self.__vehicle.control_vehicle(action)
        else:
            self.__vehicle.control_vehicle_discrete(action)

    def __timer_truncated(self):
        if time.time() - self.start_time > self.__time_limit:
            self.__time_limit_reached = True
            return True
        else:
            return False
    
    def __start_timer(self):
        self.start_time = time.time()
        
    # ===================================================== DEBUG METHODS =====================================================
    def place_spectator_above_vehicle(self):
        self.__world.place_spectator_above_location(self.__vehicle.get_location())    

    def output_all_waypoints(self, spacing=5):
        waypoints = self.__map.generate_waypoints(distance=spacing)

        for w in waypoints:
            self.__world.get_world().debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                       color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                                       persistent_lines=True)
            
    def output_waypoints_to_target(self, spacing=5):
        current_location = self.__vehicle.get_location()
        map_ = self.__map
        target_location = carla.Location(x=self.__active_scenario_dict['target_position']['x'], y=self.__active_scenario_dict['target_position']['y'], z=self.__active_scenario_dict['target_position']['z'])

        # Find the closest waypoint to the current location
        current_waypoint = map_.get_waypoint(current_location)

        # Find the closest waypoint to the target location
        target_waypoint = map_.get_waypoint(target_location)

        # Generate waypoints along the route with the specified spacing
        waypoints = []
        while current_waypoint.transform.location.distance(target_waypoint.transform.location) > spacing:
            waypoints.append(current_waypoint.transform.location)
            current_waypoint = current_waypoint.next(spacing)[0]

        # Draw the waypoints
        for w in waypoints:
            self.__world.get_world().debug.draw_string(w, 'O', draw_shadow=False,
                                                    color=carla.Color(r=255, g=0, b=0), life_time=10.0,
                                                    persistent_lines=True)