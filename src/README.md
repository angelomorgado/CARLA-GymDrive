# Carla Ultimate Template Modules

## Index

1. [Vehicle Module](#1--vehicle)
2. [Sensors](#2--sensors)
3. [Weather Control](#3--weather-control)
4. [Traffic Control](#4--traffic-controller)
5. [Map Control](#5--map-control)
6. [World](#6--world)
7. [Keyboard Control](#7--keyboard-control-module)
8. [Display](#8--display-module)
9. [Server](#9--server-module)

---
## 1- Vehicle

This module provides functionality for managing vehicles in a Carla simulation environment. It allows for the creation and destruction of vehicles, as well as the attachment of sensors defined in a JSON file. Additionally, it offers methods for controlling vehicles based on either a continuous or discrete action space.

### Attributes

#### Private

- `__vehicle (carla.Actor)`: The Carla actor representing the vehicle.
- `__sensor_dict (dict)`: A dictionary containing sensors attached to the vehicle.
- `__world (carla.World)`: The Carla world in which the vehicle exists.
- `__control (carla.VehicleControl)`: Control object for continuous vehicle control.
- `__ackermann_control (carla.VehicleAckermannControl)`: Control object for discrete vehicle control.
- `__throttle (float)`: Throttle value for continuous vehicle control.
- `__brake (float)`: Brake value for continuous vehicle control.
- `__steering_angle (float)`: Steering angle for continuous vehicle control.
- `__speed (float)`: Current speed of the vehicle in km/h.

### Methods

#### Public

- `get_vehicle()`: Get the Carla actor representing the vehicle.
- `get_location()`: Get the location of the vehicle.
- `set_autopilot(boolean)`: Set autopilot mode for the vehicle.
- `collision_occurred()`: Check if a collision has occurred.
- `lane_invasion_occurred()`: Check if a lane invasion has occurred.
- `spawn_vehicle(location=None, rotation=None)`: Spawn the vehicle in the environment.
- `get_sensor_dict()`: Get the dictionary of attached sensors.
- `destroy_vehicle()`: Destroy the vehicle and its attached sensors.
- `get_observation_data()`: Get observation data from attached sensors.
- `sensors_ready()`: Check if all attached sensors are ready.
- `change_vehicle_physics(weather_condition)`: Change vehicle physics based on weather conditions.
- `print_vehicle_physics()`: Print current vehicle physics settings.
- `control_vehicle(action)`: Control the vehicle based on a continuous action space.
- `control_vehicle_discrete(action)`: Control the vehicle based on a discrete action space.
- `toggle_lights(lights_on=True)`: Toggle vehicle lights on or off.
- `get_throttle()`: Get current throttle value.
- `get_brake()`: Get current brake value.
- `get_speed()`: Get current speed of the vehicle.

#### Private

- `__read_vehicle_file(filename)`: Read data from a JSON file.
- `__attach_sensors(vehicle_data, world)`: Attach sensors to the vehicle based on data from a JSON file.

### Vehicle Physics Customization

It is possible to customize the physics of a vehicle based on weather conditions. This can be achieved through JSON files. One such example can be found in the `test_vehicle_physics.json` file.

#### Affected Vehicle Physics by Weather Conditions Such as Rain

- **Mass** affects the vehicle's weight. A heavier vehicle may have more traction, but it may also be slower to accelerate and brake.

- **Tire friction** determines the friction between the tires and the road. Higher values result in more grip, while lower values can lead to reduced traction on slippery surfaces.
Damping Rate:

- **Damping Rate** affects the damping force applied to the wheels. It influences how quickly the wheel's vibrations are dampened. Adjusting this parameter can impact the vehicle's response on different surfaces.

- **Longitudinal Swiftness** influences how the tire responds to longitudinal forces, affecting acceleration and braking. Lower values may lead to wheel slip on slippery surfaces.

- **Drag Coefficient** influences the air resistance. While not directly related to the road surface, it can impact the overall dynamics of the vehicle, especially at higher speeds.

---
## 2- Sensors

The Sensors module provides classes for each CARLA sensor, allowing for attachment to vehicles and data retrieval using callbacks.

### Overview

This module supports various types of sensors, each serving different purposes in the simulation environment. The available sensors include:

- RGB Camera
- LiDAR
- Radar
- GNSS
- IMU
- Collision
- Lane Invasion

In addition to these sensors, there are plans for implementing future sensors such as:

- Semantic Segmentation Camera
- Instance Segmentation Camera
- Depth Camera
- Lidar Semantic Segmentation
- Obstacle Detection
- Optical Flow Camera (Motion Camera)

### Classes

#### 2.1- RGB_Camera

This class represents an RGB camera sensor.

##### Attributes

- `__sensor`: The RGB camera sensor attached to the vehicle.
- `__last_data`: The last processed image data.
- `__raw_data`: The raw image data.
- `__sensor_ready`: Flag indicating sensor readiness.

##### Methods

- `attach_rgb_camera(world, vehicle, sensor_dict)`: Attaches an RGB camera sensor to the vehicle.
- `callback(data)`: Callback function to process sensor data.
- `get_last_data()`: Retrieves the last processed image data.
- `get_data()`: Retrieves the raw image data.
- `is_ready()`: Checks if the sensor is ready.
- `destroy()`: Destroys the sensor.

#### 2.2- Lidar

This class represents a LiDAR sensor.

##### Attributes

- `__sensor`: The LiDAR sensor attached to the vehicle.
- `__last_data`: The last processed LiDAR data.
- `__raw_data`: The raw LiDAR data.
- `__sensor_ready`: Flag indicating sensor readiness.

##### Methods

- `attach_lidar(world, vehicle, sensor_dict)`: Attaches a LiDAR sensor to the vehicle.
- `callback(data)`: Callback function to process sensor data.
- `get_last_data()`: Retrieves the last processed LiDAR data.
- `get_data()`: Retrieves the raw LiDAR data.
- `is_ready()`: Checks if the sensor is ready.
- `destroy()`: Destroys the sensor.

#### 2.3- Radar

This class represents a radar sensor.

##### Attributes

- `__sensor`: The radar sensor attached to the vehicle.
- `__last_data`: The last processed radar data.
- `__raw_data`: The raw radar data.
- `__sensor_ready`: Flag indicating sensor readiness.

##### Methods

- `attach_radar(world, vehicle, sensor_dict)`: Attaches a radar sensor to the vehicle.
- `callback(data)`: Callback function to process sensor data.
- `get_last_data()`: Retrieves the last processed radar data.
- `get_data()`: Retrieves the raw radar data.
- `is_ready()`: Checks if the sensor is ready.
- `destroy()`: Destroys the sensor.

#### 2.4- GNSS

This class represents a GNSS (Global Navigation Satellite System) sensor.

##### Attributes

- `__sensor`: The GNSS sensor attached to the vehicle.
- `__last_data`: The last processed GNSS data.
- `__sensor_ready`: Flag indicating sensor readiness.

##### Methods

- `attach_gnss(world, vehicle, sensor_dict)`: Attaches a GNSS sensor to the vehicle.
- `callback(data)`: Callback function to process sensor data.
- `get_last_data()`: Retrieves the last processed GNSS data.
- `get_data()`: Retrieves the GNSS data.
- `is_ready()`: Checks if the sensor is ready.
- `destroy()`: Destroys the sensor.

#### 2.5- IMU

This class represents an Inertial Measurement Unit (IMU) sensor.

##### Attributes

- `__sensor`: The IMU sensor attached to the vehicle.
- `__sensor_ready`: Flag indicating sensor readiness.

##### Methods

- `attach_imu(world, vehicle, sensor_dict)`: Attaches an IMU sensor to the vehicle.
- `callback(data)`: Callback function to process sensor data.
- `get_last_data()`: Retrieves the last processed IMU data.
- `is_ready()`: Checks if the sensor is ready.
- `destroy()`: Destroys the sensor.

#### 2.6- Collision

This class represents a collision sensor.

##### Attributes

- `__sensor`: The collision sensor attached to the vehicle.
- `__sensor_ready`: Flag indicating sensor readiness.

##### Methods

- `attach_collision(world, vehicle, sensor_dict)`: Attaches a collision sensor to the vehicle.
- `callback(data)`: Callback function to handle collision events.
- `collision_occurred()`: Checks if a collision has occurred.
- `is_ready()`: Checks if the sensor is ready.
- `destroy()`: Destroys the sensor.

#### 2.7- Lane_Invasion

This class represents a lane invasion sensor.

##### Attributes

- `__sensor`: The lane invasion sensor attached to the vehicle.
- `__sensor_ready`: Flag indicating sensor readiness.

##### Methods

- `attach_lane_invasion(world, vehicle, sensor_dict)`: Attaches a lane invasion sensor to the vehicle.
- `callback(data)`: Callback function to handle lane invasion events.
- `lane_invasion_occurred()`: Checks if a lane invasion has occurred.
- `is_ready()`: Checks if the sensor is ready.
- `destroy()`: Destroys the sensor.

---
## 3- Weather Control

The Weather Control module provides functions to control weather conditions in the simulator.

### Overview

This module allows users to set different weather presets in the simulation environment. It includes functionality to select specific weather presets, activate random presets, and list all available presets.

#### Currently available weather presets:

- 'Clear Night'
- 'Clear Noon'
- 'Clear Sunset'
- 'Cloudy Night'
- 'Cloudy Noon'
- 'Cloudy Sunset'
- 'Default'
- 'Dust Storm'
- 'Hard Rain Night'
- 'Hard Rain Noon'
- 'Hard Rain Sunset'
- 'Mid Rain Sunset'
- 'Mid Rainy Night'
- 'Mid Rainy Noon'
- 'Soft Rain Night'
- 'Soft Rain Noon'
- 'Soft Rain Sunset'
- 'Wet Cloudy Night'
- 'Wet Cloudy Noon'
- 'Wet Cloudy Sunset'
- 'Wet Night'
- 'Wet Noon'
- 'Wet Sunset'

### Class

The WeatherControl class enables the manipulation of weather conditions in the simulation.

#### Attributes

##### Private

- `__weather_list`: List of available weather presets.
- `__active_weather`: Currently active weather preset.
- `__world`: The Carla world object.

#### Methods

##### Public

- `get_weather_presets()`: Returns a list of available weather presets.
- `get_active_weather()`: Returns the currently active weather preset.
- `print_all_weather_presets()`: Prints all available weather presets.
- `set_active_weather_preset(weather)`: Sets the active weather preset.
- `set_random_weather_preset()`: Sets a random weather preset.
- `choose_weather()`: Allows the user to choose a weather preset.

##### Private

- `__get_all_weather_presets()`: Retrieves all available weather presets.
- `__activate_weather_preset(idx)`: Activates a specific weather preset.


---
## 4- Traffic Controller

The Traffic Controller module provides functionality to spawn, destroy, and control vehicles and pedestrians in the Carla simulation.

### Overview

This module enables users to manipulate traffic elements within the simulation environment, including vehicles and pedestrians.

### Class

The TrafficControl class manages the spawning, control, and destruction of vehicles and pedestrians.

#### Attributes

##### Private

- `__active_vehicles`: List of currently active vehicle actors.
- `__active_pedestrians`: List of currently active pedestrian actors.
- `__active_ai_controllers`: List of active AI controllers for pedestrians.
- `__world`: The Carla world object.
- `__map`: The Carla map object.

#### Methods

##### Public

- `update_map(map)`: Updates the map used by the traffic controller.
- `spawn_vehicles(num_vehicles=10, autopilot_on=False)`: Spawns vehicles in the simulation.
- `destroy_vehicles()`: Destroys all active vehicles.
- `toggle_autopilot(autopilot_on=True)`: Toggles autopilot mode for vehicles.
- `spawn_vehicles_around_ego(ego_vehicle, radius, num_vehicles_around_ego, seed=None)`: Spawns vehicles around the ego vehicle within a specified radius.
- `toggle_lights(lights_on=True)`: Toggles vehicle lights on or off.
- `spawn_pedestrians(num_walkers=10)`: Spawns pedestrians on random sidewalks.    
- `spawn_pedestrians_around_ego(vehicle_location, num_walkers=10, radius=25.0)`: Spawns pedestrians around the ego vehicle within a specified radius.
- `destroy_pedestrians()`: Destroys all active pedestrians.

---
## 5- Map Control

The Map Control module manages the current map of the simulation and allows for its customization.

### Overview

The Map Control module provides functionalities to control the map used in the simulation environment and enables customization options.

### Class

The MapControl class controls the current map of the simulation and allows for its customization.

#### Attributes

##### Private

- `__world`: The Carla world object.
- `__client`: The Carla client object.
- `__available_maps`: List of available maps in the simulation environment.
- `__map_dict`: Dictionary mapping map names to their corresponding indices.
- `__active_map`: Index of the currently active map.
- `__map`: The current map object.

#### Methods

##### Public

- `get_active_map_name()`: Returns the name of the currently active map.
- `get_map()`: Returns the current map object.
- `print_available_maps()`: Prints the available maps in the simulation environment.
- `set_active_map(map_name, reload_map=False)`: Sets the active map to the specified map name. If reload_map is True, reloads the map.
- `change_map()`: Allows the user to choose and change the active map (for debugging purposes).
- `reload_map()`: Reloads the current active map.

---
## 6- World

The World module serves as a compilation of various other modules for easier use inside a script. Instead of importing multiple modules separately, one can simply import this module.

### Overview

The World module provides a unified interface for managing different aspects of the simulation environment in Carla. It includes functionalities related to traffic control, weather control, map control, and more.

### Available Modules

- Traffic
- Weather Control
- Map
- Spectator (This one isn't in a different module because it's just two simple functions)

### Class

The World class serves as the main interface for controlling different aspects of the Carla simulation environment.

#### Attributes

##### Private

- `__client`: The Carla client object.
- `__world`: The Carla world object.
- `__weather_control`: Instance of the WeatherControl class for managing weather in the simulation.
- `__traffic_control`: Instance of the TrafficControl class for managing vehicles and pedestrians.
- `__map_control`: Instance of the MapControl class for managing the current map of the simulation.
- `__synchronous_mode`: Flag indicating whether the simulation is running in synchronous mode.

#### Methods

##### Public

- `get_client()`: Returns the Carla client object.
- `get_world()`: Returns the Carla world object.
- `destroy_world()`: Destroys all vehicles and pedestrians in the simulation.
- `tick()`: Advances the simulation by one tick.
- `get_weather_presets()`: Returns a list of available weather presets.
- `print_all_weather_presets()`: Prints all available weather presets.
- `set_active_weather_preset(weather)`: Sets the active weather preset.
- `choose_weather()`: Allows the user to choose the active weather preset.
- `get_active_weather()`: Returns the active weather preset.
- `get_active_map_name()`: Returns the name of the currently active map.
- `get_map()`: Returns the current map object.
- `print_available_maps()`: Prints all available maps.
- `set_active_map(map_name, reload_map=False)`: Sets the active map.
- `change_map()`: Allows the user to choose and change the active map (for debugging purposes).
- `reload_map()`: Reloads the current active map.
- `spawn_vehicles(num_vehicles=10, autopilot_on=False)`: Spawns vehicles in the simulation.
- `spawn_vehicles_around_ego(ego_vehicle, radius, num_vehicles_around_ego, seed=None)`: Spawns vehicles around the ego vehicle.
- `destroy_vehicles()`: Destroys all vehicles in the simulation.
- `toggle_autopilot(autopilot_on=True)`: Toggles autopilot mode for vehicles.
- `spawn_pedestrians(num_pedestrians=10)`: Spawns pedestrians in the simulation.
- `spawn_pedestrians_around_ego(ego_vehicle_location, num_pedestrians=10, radius=50)`: Spawns pedestrians around the ego vehicle.
- `destroy_pedestrians()`: Destroys all pedestrians in the simulation.
- `toggle_lights(lights_on=True)`: Toggles vehicle lights.
- `update_traffic_map()`: Updates the traffic map.
- `place_spectator_above_location(location)`: Places the spectator camera above a specified location.
- `place_spectator_behind_location(location, rotation)`: Places the spectator camera behind a specified location with the given rotation.

---
## 7- Keyboard Control Module

The Keyboard Control Module provides the functionality to control a vehicle using the keyboard.

### Overview

This module enables users to control vehicle movement using specific keys on the keyboard. It includes functionality for throttle, brake, steering, reversing, and locking/unlocking vehicle control.

### Keys

- **w**: Throttle
- **s**: Brake
- **a**: Steer left
- **d**: Steer right
- **q**: Toggle reverse
- **z**: Toggle lock (toggles vehicle control on/off)

### Class

The KeyboardControl class manages the keyboard inputs and applies corresponding controls to the vehicle.

#### Attributes

##### Private

- `__vehicle`: The vehicle object to control.
- `__throttle`: Throttle value (0.0 to 1.0).
- `__brake`: Brake value (0.0 to 1.0).
- `__steering`: Steering value (-1.0 to 1.0).
- `__reverse`: Boolean indicating whether the vehicle is in reverse mode.
- `__lock`: Boolean indicating whether vehicle control is locked.
- `__listener`: Keyboard listener object.

#### Methods

##### Private

- `__on_press(key)`: Callback function for key press events.
- `__on_release(key)`: Callback function for key release events.

##### Public

- `apply_controls()`: Applies the current control values to the vehicle.
- `tick()`: Applies controls to the vehicle (main update function).
- `clean()`: Stops and joins the keyboard listener.

---
## 8- Display Module

The Display Module provides functionality to display sensor data in a window using Pygame.

### Overview

This module enables users to visualize sensor data such as camera images, GNSS (Global Navigation Satellite System) data, and IMU (Inertial Measurement Unit) data in a Pygame window. It includes methods to create and manage the display window, as well as to update and close it.

### Class

The Display class manages the creation and updating of the Pygame window to display sensor data.

#### Attributes

##### Private

- `__sensor_window_dict`: A dictionary to store sensor data surfaces.
- `__sensor_dict`: A dictionary containing references to the vehicle's sensors.
- `__non_displayable_sensors`: A list of sensor types that are not displayed.
- `__main_screen`: The main Pygame window surface.
- `__clock`: A Pygame clock object to control the frame rate.

#### Methods

##### Public

- `initialize_pygame_window(title)`: Initializes the Pygame window with the given title.
- `play_window()`: Displays the sensor data in a Pygame window using its own event loop.
- `play_window_tick()`: Displays the sensor data in a Pygame window inside the main loop of the program.
- `close_window()`: Closes the Pygame window.

---
## 9- Server Module

The Server Module contains the CarlaServer class responsible for starting and stopping the Carla server.

### Overview

This module facilitates the management of the Carla server, including its initialization, shutdown, and termination. It provides methods to start the server, close it gracefully, and forcibly terminate it, depending on the operating system.

### Class

The CarlaServer class encapsulates functionalities related to the Carla server.

#### Methods

##### Static Methods

- `initialize_server(low_quality=False, offscreen_rendering=False, silent=False, sleep_time=10)`: Initializes the Carla server with optional parameters such as quality level and offscreen rendering. It waits for the server to start before returning a process object representing the server.
- `close_server(process, silent=False)`: Gracefully closes the Carla server. On Unix systems, it sends a termination signal to the process group. On Windows, it forcibly terminates the process and its children.
- `kill_carla_linux()`: Terminates the Carla server forcefully on Unix systems by killing the process using the `pkill` command. This method is not applicable to Windows systems.
