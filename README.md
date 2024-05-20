# CARLA GymDrive

Carla GymDrive is a powerful framework designed to facilitate reinforcement learning experiments in autonomous driving using the Carla simulator. By providing a gymnasium-like environment, it offers an intuitive and efficient platform for training driving agents using reinforcement learning techniques.

## Citation

If you use Carla GymDrive in your research, please cite using the following citation:

**BibTeX:**
```bibtex
@misc{2024CARLAGymDrive,
  title = {CARLA-GymDrive: Autonomous driving episode generation for the Carla simulator in a gym environment.},
  author = {Ângelo Morgado and Nuno Pombo},
  year = {2024},
  note = {Contact email: angelo.morgado@ubi.pt}
}
```

**APA:**
```apa
Morgado, Â., & Pombo, N. (2024). CARLA-GymDrive: Autonomous driving episode generation for the Carla simulator in a gym environment. Contact email: angelo.morgado@ubi.pt
```

## Features

### Seamless Integration and Easy Customization

Carla GymDrive seamlessly integrates with the Carla simulator, allowing users to leverage its extensive features for creating realistic driving scenarios. It also provides an easy to customize environment for fine-tuning the simulation parameters to the user's needs.

### Reinforcement Learning Ready

With built-in compatibility for reinforcement learning libraries such as Stable Baselines3, Carla GymDrive streamlines the process of training autonomous driving agents.

### Modular Design

The framework is built with a modular design approach, enabling easy customization and integration into various projects and workflows.

### Custom Vehicular Sensory

By leveraging json files, it is possible to create various builds of vehicles with different sensors and configurations. This allows for the creation of custom vehicles with different sensor configurations. Such example of a build can be found in the [./env/train_sensors.json](./env/train_sensors.json) file.

### Sensor Visualization

Through Pygame, it is possible to visualize the sensor data in real-time. This is useful for debugging and testing purposes.

### Vehicle Physics Customization

Vehicles have their physics changed according to the weather. This template allows for the customization of a vehicle's physics based on the weather conditions. This is useful for simulating the effects of weather on a vehicle's performance. This can be achieved through JSON files. One such example can be found in the [./env/default_vehicle_physics.json](./env/default_vehicle_physics.json) file.

### Complete Simulation Control and Management Using Minimal Code

The template allows for the complete control and management of the Carla simulator using minimal code. This is achieved through the use of the `World` class. This class allows for the easy management of the Carla simulator, such as changing the map, the weather, and even spawning traffic and pedestrians.

---

## Installation and Usage

1. It is recommended to use a virtual environment with python 3.8.

 - If you wish to use the same virtual environment as me which used Carla 0.9.15, install conda and run `conda env create -f environment.yml`
 - If you wish to use a different version of Carla, you can create a new environment with `conda create -n carla python=3.8` and then install the requirements with `pip install -r requirements.txt`

3. Setup the environment variable `CARLA_SERVER` to the path of the Carla server directory.

4. Run the Carla server:

 - If the script automatically starts the server, you can skip this step. Make sure to set the environment variable `CARLA_SERVER` to the path of the Carla server directory.
 - If the script does not automatically start the server, you need to start the server manually.

5. Run training/testing scripts

---

## Episode Generation

For a more extensive documentation on how to generate episodes, please refer to the [Environment](env/README.md) documentation.

---

## Modules

This template's modules are located and documented in the `src` directory. Their documentation can be found [here](src/README.md)

---

## Known Issues

- The simulator may crash when changing maps to many times. This is a known issue with Carla and is not a problem with the template. The problem is random, so if it happens, it is recommended to save checkpoints and then restart the training in the latest checkpoint.
- If the simulator is ran in low quality mode, it crashes the program, this is a problem in Carla's side and it's known by the community;
- Moving the walkers causes segmentation fault. This is a known problem between the community.
- Simply spawning the walkers might cause the program to crash. This is maybe due to my personal computer's performance. I haven't tested it in a more powerfull pc. I don't think it is a coding problem, but i might be wrong.

---

## License

Carla GymDrive is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.

## Acknowledgements

Carla GymDrive is inspired by the open-source community and contributions from researchers and developers around the world. We would like to express our gratitude to the Carla team for providing an excellent simulator for autonomous driving research.
