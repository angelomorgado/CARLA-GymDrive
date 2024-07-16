# Carla Environment

This directory contains the code to configure the Carla environment for the Reinforcement Learning tasks.

## Instructions Manual

There are two ways of using the environment:

1. The first, and the recommended one is through the gym framework. An example of the main loop is as such:

    ```python
    import gymnasium as gym
    import src.env.environment

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
    import gymnasium as gym
    from src.env.environment import CarlaEnv

    env = CarlaEnv()
    # ...
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
- `apply_physics` (bool): If True, it applies the physics in the physics file to the simulation. If False, the default physics are maintained through all weather conditions.
- `autopilot` (bool): If True, the ego vehicle is controlled by the autopilot. If False, the ego vehicle is controlled by the agent. It is recommended to give an action that doesn't move the vehicle. Its main usage is for debugging purposes or even demonstration purposes.
- `verbose` (bool): If True, it displays more detailed outputs about the episodes.

### Scenario customization

One of the main advantages of this framework is the ability to easily customize the training/testing scenarios. More information about scenario suite customization can be found in the [configuration documentation](../config/README.md). 

### Observation Space Customization

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

To  change the observation space you need to change the file [observation_action_space.py](../env/observation_action_space.py); and then go to the [CarlaEnv](../env/environment.py) class and change the `__update_observation` method.

### Action Space Customization

Observation space is totally customizable, and it follows the gymnasium.Spaces standard, however, if you wish to use the default ones, the observation space is:

```python!

# For continuous actions
# [Steering (-1.0, 1.0), Throttle/Brake (-1.0, 1.0)] <- [-1, 0] brake / [0, 1] throttle
self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

# For discrete actions
# 0: Accelerate, 1: Deaccelerate, 2: Turn Left, 3: Turn Right
self.action_space = spaces.Discrete(4)
```

To  change the action space you need to change the file [observation_action_space.py](../env/observation_action_space.py).

### Reward Function Customization

The reward function is fully customizable. To finetune it you can simply change the function `calculate_reward` in the file [reward.py](../env/reward.py). If you want to change the signature of the function, in case you need additional data to calculate the reward, don't forget to also change it in the [CarlaEnv](../env/environment.py) class!

### Methods

The public methods accessible through the CarlaEnv class are:

#### Standard gym methods

- `env.reset()`: Starts a new episode in a random scenario.
  - seed: Seed to make the episode deterministic
  - options: Dictionary with the key `scenario_name` to specify the specific scenario to load in case the problem requires it.
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
- `env.output_all_waypoints(spacing)`: Outputs on the server screen all waypoints of the map separated by a determined spacing.
- `env.draw_waypoints(waypoint_list, life_time)`: Outputs on the server screen the waypoints present in the provided list.
- `env.get_path_waypoints(spacing)`: Returns a list of waypoints of the scenario path separated by a determined spacing.

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
