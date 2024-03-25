# Episode Generation

## Introduction

The Carla Episode Generator is a tool designed to facilitate the creation of diverse training and testing scenarios for Reinforcement Learning (RL) models in the Carla Simulator [src](https://carla.org/). This tool allows the definition of structured scenarios, the control of environmental parameters such as the map, weather conditions, and traffic density.

This tool was designed having Carla 0.9.15 in mind, however it should work with other versions.

The tool is a gym.Env wrap. This means that you can use this environment as a gym environment and it is even compatible with gym-compatible frameworks for training/testing such as stable-baselines3.

## Instalation

1. It is recommended to use a virtual environment with python 3.8 (e.g., conda).
2. Set the environmental variable `CARLA_SERVER` as the location of the Carla server's directory.
3. Install the requirements with pip install -r requirements.txt.
4. (Optional) Run the CARLA server (It is not recommended to launch the server in low-quality mode as it can cause crashes). If you want the server to start and close automatically you can use the flag `initialize_server=True` when instaciating the environment.
5. Run the client script (e.g., train or test script).

## TODO

- [ ] Put the reward functions in their own file for easier customization.
- [ ] Put the observation space and action space in their own file for easier customization.
- [ ] Make code more clean and consistent (better organization).
- [ ] Add more training parameters.

## Known Issues

- If the simulator is ran in low quality mode, it crashes the program, this is a problem in Carla's side and it's known by the community;
- If the first episode's map is the same as the simulators map, it loads everything so quickly that the sensors' threads don't send the sensors' data quick enough and the program crashes (I'll try to fix this ASAP (Also if you have ideas on how to fix it be sure to contact me.));
- Moving the walkers causes segmentation fault. This is a known problem between the community.
- Simply spawning the walkers might cause the program to crash. This is maybe due to my personal computer's performance. I haven't tested it in a more powerfull pc. I don't think it is a coding problem, but i might be wrong.

## Instructions Manual

There are two ways of using the environment:

1. The first, and the recommended one is through the gym framework. An example of the main loop is as such:

    ```python
    import gymnasium as gym
    import env.environment

    def steps_main():
        env = gym.make('carla-rl-gym-v0')
        obs, info = env.reset()

        for i in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()
    ```

2. The second, is through the CarlaEnv class:

    ```python
    from env.environment import CarlaEnv

    def steps_main():
        env = CarlaEnv()
        obs, info = env.reset()

        for i in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()
    ```

The environment's constructor has multiple arguments for the customization of the environment, these are:

- `continuous` (bool): Determines if the action space is continuous (True) or discrete (False);
- `scenarios` (list: Road/Roundabout,etc.): List of desired scenarios if you don't want to segmentate the scenarios JSON.
- `time_limit` (int): Maximum amount of seconds for each episode. When it reaches this timeout the episode gets truncated.
- `initialize_server` (bool): Automatically opens and closes the server. If False, you have to open the server side before running the client side scripts;
- `random_weather` (bool): If True loads a random weather configuration for each episode regardless of what's in the scenarios JSON. If False, simply loads what's in the JSON.
- `random_traffic` (bool): If True loads a random traffic configuration for each episode regardless of what's in the scenarios JSON. If False, it loads the traffic based on the scenario's name. It can be overwritten if given a seed to the reset function.
- `synchronous_mode` (bool): If True loads the client in synchronous mode. It is recommended to keep this as True, as some Carla features require it to be on.
- `show_sensor_data` (bool): If True, during each episode it opens up a pygame window with the ego vehicle's sensors for easy visualization.
- `has_traffic` (bool): If False, it loads the episodes without any traffic at all.
- `verbose` (bool): If True, it displays more detailed outputs about the episodes.

## Simulation configuration

### Configuration

In the file [configuration.py](../configuration.py) there are multiple different configuration options, regarding basically everything about the program. Note that there is a Verbose option there, however it is different from the verbose argument of the constructor, as it prints almost everything that the program does, so it's more powerful.

### Ego Vehicle's Sensors Configuration

To change the ego vehicle's sensors and even their paramenters there's an easy way to do so, simply create a JSON file and make something along the lines of:

```python
{
    "rgb_camera":{
        "image_size_x": 640,
        "image_size_y": 360,
        "fov": 110,
        "sensor_tick": 0.0,
        "location_x": 0.8,
        "location_y": 0.0,
        "location_z": 1.7
    },
    "lidar":{
        "channels": 64,
        "range": 100.0,
        "points_per_second": 1000000,
        "rotation_frequency": 50.0,
        "upper_fov": 20.0,
        "lower_fov": -30.0,
        "sensor_tick": 0.0,
        "location_x": 0.8,
        "location_y": 0.0,
        "location_z": 1.7
    },
    "collision":{
        "location_x": 0.0,
        "location_y": 0.0,
        "location_z": 0.0
    },
    "lane_invasion":{
        "location_x": 0.0,
        "location_y": 0.0,
        "location_z": 0.0
    }
}
```

Note that it is important to follow the naming standard or else the program might not work as it is expecting certain names and they do not exist.

A list of available sensors can be found [here](../README.md).

### Scenario Customization

Making new scenarios or changing the existing ones is super intuitive. Basically you only have to create a JSON file to manage the scenarios. Then you have to specify the path to the json in the configuration.py file.

It must specify the following parameters:

- `map_name`: The name of the CARLA map. Don't specify the path, simply the name of the map
- `weather_condition`: The weather conditions (e.g., "ClearNoon").
- `initial_position`: Initial position of the ego vehicle (x, y, z coordinates).
- `initial_rotation`: Initial rotation of the ego vehicle (pitch, yaw, roll)
- `situation`: Type of scenario or situation (e.g., "Road", "Roundabout," "Junction" and "Tunnel")
- `target_position`: Target position for the ego vehicle

An example of a JSON file is below:

```json
{
  "Town01-ClearNoon-Road-0": {
    "map_name": "Town01",
    "weather_condition": "Clear Noon",
    "initial_position": {"x": 312.3, "y": 195.3, "z": 0.3},
    "initial_rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
    "target_position": {"x": 128.2, "y": 195.3, "z": 0.3},
    "target_gnss": {"lat": -0.001754, "lon": 0.001143, "alt": 0},
    "situation": "Road"
  },
  "Town01-ClearNoon-Junction-0": {
    "map_name": "Town01",
    "weather_condition": "Clear Noon",
    "initial_position": {"x": 51.4, "y": 330.8, "z": 0.3},
    "initial_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    "target_position": {"x": 128.1, "y": 330.8, "z": 0.3},
    "target_gnss": {"lat": -0.002972, "lon": 0.001160, "alt": 0},
    "situation": "Junction"
  },
  "Town01-ClearNoon-Junction-1": {
    "map_name": "Town01",
    "weather_condition": "Clear Noon",
    "initial_position": {"x": 51.4, "y": 330.8, "z": 0.3},
    "initial_rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
    "target_position": {"x": 92.6, "y": 302.1, "z": 0.3},
    "target_gnss": {"lat": -0.002705, "lon": 0.000832, "alt": 0},
    "situation": "Junction"
  },
  "Town10HD-ClearNoon-Road-0": {
    "map_name": "Town10HD",
    "weather_condition": "Clear Noon",
    "initial_position": {"x": 59.5, "y": 130.5, "z": 0.3},
    "initial_rotation": {"pitch": 0.0, "yaw": 180.0, "roll": 0.0},
    "target_position": {"x": -15.3, "y": 129.9, "z": 0.3},
    "target_gnss": {"lat": -0.001167, "lon": -0.000146, "alt": 0},
    "situation": "Junction"
  },
....
}
```

### Observation Space

Observation space is totally customizable, and it follows the gymnasium.Spaces standard, however, if you wish to use the default ones, the observation space is:

```python
self.rgb_image_shape = (360, 640, 3)
self.lidar_point_cloud_shape = (500,4)
self.current_position_shape = (3,)
self.target_position_shape = (3,)
self.number_of_situations = 4

self.observation_space = spaces.Dict({
            'rgb_data':        spaces.Box(low=0, high=255, shape=self.rgb_image_shape, dtype=np.uint8),
            'lidar_data':      spaces.Box(low=-np.inf, high=np.inf, shape=self.lidar_point_cloud_shape, dtype=np.float32),
            'position':        spaces.Box(low=-np.inf, high=np.inf, shape=self.current_position_shape, dtype=np.float32),
            'target_position': spaces.Box(low=-np.inf, high=np.inf, shape=self.target_position_shape, dtype=np.float32),
            'situation':       spaces.Discrete(self.number_of_situations) # *
        })
```

\* The Situations are: 0: Road, 1: Roundabout, 2: Junction, 3: Tunnel
However you can customize this by changing the dictionary and the number of situations variable.

To  change the observation space you can do it at the file [observation_action_space.py](../env/observation_action_space.py).

### Action Space

Observation space is totally customizable, and it follows the gymnasium.Spaces standard, however, if you wish to use the default ones, the observation space is:

```python!

# For continuous actions
# [Steering (-1.0, 1.0), Throttle/Brake (-1.0, 1.0)] <- [-1, 0] brake / [0, 1] throttle
self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

# For discrete actions
# 0: Accelerate, 1: Deaccelerate, 2: Turn Left, 3: Turn Right
self.action_space = spaces.Discrete(4)
```

### Reward Function

To customize the reward function you can simply change the function `calculate_reward` in the file [reward.py](../env/reward.py). If you want to change the signature of the function, don't forget to also change it in the [CarlaEnv](../env/environment.py) class!

The default reward function takes into account these factors:
- The orientation of the ego vehicle. To do this it uses the cousine of the angle between the ego vehicle's forward vector and the road's forward vector. The closer to 1, the better.
- The distance between the ego vehicle and the waypoint location. The closer, the better. (I'm thinking in removing this one)
- The speed of the ego vehicle. It it passes the limit speed, it gets a penalty.
- It penalizes the car if it's stopped.
- It ends the simulation and penalizes severely the ego vehicle if it has a collision, trespasses a lane, or goes off-road.
- It ends the simulation and penalizes severely the ego vehicle if it doesn't stop at a red light or at a stop sign.

### Methods

The public methods accessible through the CarlaEnv class are:

#### Standard gym methods

- `env.reset()`: Starts a new episode in a random scenario.
  - seed: Seed to make the episode deterministic
  - options: extra options. There are none at the moment
- `env.step(action)`: Takes a step in the environment. The action must be according the action space.
- `env.render()`: Ticks the simulation
- `env.close()`: Closes the simulation

#### Episode methods

- `env.load_scenario(scenario_name, seed)`: Loads the scenario, at the moment the seed is mandatory.
- `env.clean_scenario()`: Cleans the scenario without changing map nor closing the simulation.
- `env.print_all_scenarios()`: Outputs the name of every scenario available.
- `env.load_world(map_name)`: Loads a map by its name. It does the same as the World module's set_active_map().

#### Debug methods

- `env.place_spectator_above_vehicle()`: It places the server screen on top of the ego vehicle.
- `env.output_all_waypoints(spacing)`: Outputs on the server screen all waypoints separated by a determined spacing.
- `env.output_waypoints_to_target(spacing)`: Outputs on the server screen the waypoints from the starting point of the scenario to the target point, separated by a certain spacing.

## Attributes

The name of the scenario will be based on these attributes, then because there can be many there will be a counter after the name to differentiate.

Available Maps:

- Town01 <- small city roads with junctions with lightpoles
- Town02 <- Has city roads and junctions (neighborhood)
- Town03 <- Big city with roundabouts lightpoles, big junctions and inclined roads, also has big road and even a tunnel.
- Town04 <- Highway with small city in the center with lightpoles and junctions
- Town05 <- Highway with big city in the center with big junctions, lightpoles
- Town07 <- Village road with street signs and small intersections with lightpoles
- Town10HD <- Default map, big city with intersections, lightpoles
- Town15 <- This is the biggest city and it has every type of scenario possible, it’s used for testing the carla leaderboard challenge
Each map has a variation with the suffix “_opt”. This means that the map is layered and different layers can be removed and added with scripts. But these won't be used.

(Towns 11-13 make my machine crash, you can try them, i added compatibility to them but they still crash in my machine, beware)

Available Weather Conditions

- Clear Night
- Clear Noon
- Clear Sunset
- Cloudy Night
- Cloudy Noon
- Cloudy Sunset
- Default
- Dust Storm
- Hard Rain Night
- Hard Rain Noon
- Hard Rain Sunset
- Mid Rain Sunset
- Mid Rainy Night
- Mid Rainy Noon
- Soft Rain Night
- Soft Rain Noon
- Soft Rain Sunset
- Wet Cloudy Night
- Wet Cloudy Noon
- Wet Cloudy Sunset
- Wet Night
- Wet Noon
- Wet Sunset

Available Traffic Densities

- None
- Low
- High
