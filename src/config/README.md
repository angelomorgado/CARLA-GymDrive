# Simulation configuration

This directory contains all the configuration files for the simulation.

## Simulation Configuration

In the file [configuration.py](./configuration.py) there are multiple different configuration options, regarding basically everything about the simulation. Note that there is a Verbose option there, however it is different from the verbose argument of the constructor, as it prints almost everything that the program does, so it's more powerful.

The configuration options are:

- `IM_WIDTH`: The width of the pygame window
- `IM_HEIGHT`: The height of the pygame window
- `NUM_COLS`: Number of columns in the grid
- `NUM_ROWS`: Number of rows in the grid
- `MARGIN`: Margin between the grid cells
- `BORDER_WIDTH`: Width of the border of the grid cells
- `SENSOR_FPS`: The FPS of the sensors
- `VERBOSE`: If True, it prints a lot of information about the simulation
- `VEHICLE_SENSORS_FILE`: The path to the JSON file with the sensors configuration
- `VEHICLE_PHYSICS_FILE`: The path to the JSON file with the vehicle physics configuration
- `VEHICLE_MODEL`: The model of the vehicle
- `SIM_HOST`: The host of the simulation
- `SIM_PORT`: The port of the simulation
- `SIM_TIMEOUT`: The timeout of the simulation
- `SIM_LOW_QUALITY`: If True, it runs the simulation in low quality
- `SIM_OFFSCREEN_RENDERING`: If True, it runs the simulation in offscreen rendering
- `SIM_FPS`: The FPS of the simulation
- `ENV_SCENARIOS_FILE`: The path to the JSON file with the scenarios configuration
- `ENV_MAX_STEPS`: The maximum number of steps per episode
- `ENV_WAYPOINT_SPACING`: The spacing of the waypoints

## Ego Vehicle's Sensors Configuration

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

Note that it is important to follow the naming standard or else the program might not work as it is expecting certain names and they do not exist. An example of a sensor file can be found in the [default_sensors.json](./default_sensors.json) file.

A list of available sensors can be found [here](../carlacore/README.md).

## Scenario Customization

Making new scenarios or changing the existing ones is super intuitive. Basically you only have to create a JSON file to manage the scenarios. Then you have to specify the path to the json in the configuration.py file.

It must specify the following parameters:

- `map_name`: The name of the CARLA map. Don't specify the path, simply the name of the map
- `weather_condition`: The weather conditions (e.g., "ClearNoon").
- `initial_position`: Initial position of the ego vehicle (x, y, z coordinates).
- `initial_rotation`: Initial rotation of the ego vehicle (pitch, yaw, roll)
- `situation`: Type of scenario or situation (e.g., "Road", "Roundabout," "Junction" and "Tunnel")
- `target_position`: Target position for the ego vehicle

An example of a scenario is as follows:

```json
{
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

An example of a scenario file can be found in the [default_scenarios.json](./default_scenarios.json) file.

## Vehicle Physics Configuration

In some problems it is necessary to change the vehicle's physics based on the weather. This can be achieved through JSON files. An example of a build can be found in the [default_vehicle_physics.json](./default_vehicle_physics.json) file.