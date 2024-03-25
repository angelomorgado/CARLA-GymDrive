# CARLA Ultimate Template

This project acts as a template for the Carla Simulator. It is a collection of various features and functionalities that can be used to create a custom environment for the Carla Simulator. More features will be added as the project progresses.
This project can be seen as an engine of sorts, and acts as a starting point for creating various scenarios for the Carla Simulator with relative ease as all the necessary components are already in place.

All the modules are designed to be as modular as possible, so that they can be easily integrated into other projects and they are organized in classes and functions.

```
I'll make a proper documentation and README once the project is finished.
```

---

## Installation and Usage

1. It is recommended to use a virtual environment with python 3.8.
 - If you wish to use the same virtual environment as me which used Carla 0.9.15, install conda and run `conda env create -f environment.yml`
 - If you wish to use a different version of Carla, you can create a new environment with `conda create -n carla python=3.8` and then install the requirements with `pip install -r requirements.txt`
3. Setup the environment variable `CARLA_SERVER` to the path of the Carla server directory.
4. Run the Carla server:
 - If the script automatically starts the server, you can skip this step. Make sure to set the environment variable `CARLA_SERVER` to the path of the Carla server directory.
 - If the script does not automatically start the server, you need to start the server manually.
5. Run any scripts

---

## Modules

This template's modules are located and documented in the `src` directory. Their documentation can be found [here](src/README.md)

---

## Features

### Gymnasium Environment

Using this template, a gymnasium wrapper was created for the Carla simulator. This allows the training of autonomous driving agents using reinforcement learning algorithms. By wrapping the Carla simulator in a gym environment, it is possible to use libraries such as Stable Baselines to train agents.

More about this tool can be found in [its documentation](env/README.md)

### Custom Vehicular Sensory

By leveraging json files, it is possible to create various builds of vehicles with different sensors and configurations. This allows for the creation of custom vehicles with different sensor configurations. Such example of a build can be found in the `test_sensors.json` file.

### Sensor Visualization

Through Pygame, it is possible to visualize the sensor data in real-time. This is useful for debugging and testing purposes.

### Vehicle Physics Customization

Vehicles have their physics changed according to the weather. This template allows for the customization of a vehicle's physics based on the weather conditions. This is useful for simulating the effects of weather on a vehicle's performance. This can be achieved through JSON files. One such example can be found in the `test_vehicle_physics.json` file.

### Complete Simulation Control and Management Using Minimal Code

The template allows for the complete control and management of the Carla simulator using minimal code. This is achieved through the use of the `World` class. This class allows for the easy management of the Carla simulator, such as changing the map, the weather, and even spawning traffic and pedestrians.
