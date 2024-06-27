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
  url = {https://github.com/angelomorgado/CARLA-GymDrive},
  note = {Contact: angelo.morgado@ubi.pt}
}
```

**APA:**
```apa
Morgado, Â., \& Pombo, N. (2024). CARLA-GymDrive: Autonomous driving episode generation for the Carla simulator in a gym environment. email: angelo.morgado@ubi.pt. [Online]. Available: \url{https://github.com/angelomorgado/CARLA-GymDrive}.
```

---

## Project Structure

The project is structured as follows:

- `src`: Contains the source code for the Carla GymDrive framework.
- [`src/env`](src/env/README.md): Contains the environment code for the Carla GymDrive framework, as well as the files for observation/action space and reward function customization.
- [`src/config`](src/config/README.md): Contains the configuration files for the Carla GymDrive framework, such as sensor configurations, scenario configurations, and vehicle physics configurations.
- [`src/carlacore`](src/carlacore/README.md): Contains the back-end code that acts as an interface between the Carla simulator and the environment.	

## Main Features

### Seamless Integration and Easy Customization

Carla GymDrive seamlessly integrates with the Carla simulator, allowing users to leverage its extensive features for creating realistic driving scenarios. It also provides an easy to customize environment for fine-tuning the simulation parameters to the user's needs.

### Reinforcement Learning Ready

With built-in compatibility for reinforcement learning libraries such as Stable Baselines3, Carla GymDrive streamlines the process of training autonomous driving agents.

### Modular Design

The framework is built with a modular design approach, enabling easy customization and integration into various projects and workflows.

### Custom Vehicular Sensory

By leveraging json files, it is possible to create various builds of vehicles with different sensors and configurations. This allows for the creation of custom vehicles with different sensor configurations.

### Sensor Visualization

Through Pygame, it is possible to visualize the sensor data in real-time. This is useful for debugging and testing purposes.

### Vehicle Physics Customization

Vehicles have their physics changed according to the weather. This template allows for the customization of a vehicle's physics based on the weather conditions. This is useful for simulating the effects of weather on a vehicle's performance. This can be achieved through JSON files.

### Complete Simulation Control and Management Using Minimal Code

The template allows for the complete control and management of the Carla simulator using minimal code. This is achieved through the use of the `World` class. This class allows for the easy management of the Carla simulator, such as changing the map, the weather, and even spawning traffic and pedestrians.

---

## Installation and Usage

1. It is recommended to use a virtual environment with python 3.8.

- If you wish to use the same virtual environment as me which used Carla 0.9.15, install conda and run `conda env create -f environment.yml`
- If you wish to use a different version of Carla, you can create a new environment with `conda create -n carla python=3.8` and then install the requirements with `pip install -r requirements.txt`

2. Setup the environment variable `CARLA_SERVER` to the path of the Carla server directory.

3. Run the Carla server:

- If the script automatically starts the server, you can skip this step. Make sure to set the environment variable `CARLA_SERVER` to the path of the Carla server directory.
- If the script does not automatically start the server, you need to start the server manually.

4. Run training/testing scripts.

---

## Environment Configuration

There are countless options for configuring the simulation and the gym environment. In order to fine-tune the environment to your needs, you can change the following parameters in the [`src/config/configuration.py`](`src/config/configuration.py`) file.

---

## Known Issues

- The simulator may crash when changing maps to many times. This is a known issue with Carla and is not a problem with the template. The problem is random, so if it happens, it is recommended to save checkpoints and then restart the training in the latest checkpoint. Example of the issue being reported [here](https://github.com/carla-simulator/carla/issues/4711);
- If the simulator is ran in low quality mode, it crashes the program, this is a problem in Carla's side and it's known by the community. Issue reported [here](https://github.com/carla-simulator/carla/issues/6399);
- Moving the walkers causes segmentation fault. This is a known problem between the community[here](https://github.com/carla-simulator/carla/issues/4155);

---
## Helpful Scripts

The directory `helpful-scripts` contains some useful scripts for using the this environment not only for a development purpose but also for a research purpose.

## Agent Training

This repository doesn't contain any agent custom policies, however it provides an example training script for the DQN algorithm using the stable-baselines3 library. The script name is `example_sb3_dqn_training.py`.

If you want to see the agents I used in my thesis research, you can check the [CARLA-RL-Agents repository](https://github.com/angelomorgado/CARLA-RL-Agents).

---

## License

Carla GymDrive is licensed under the MIT License. See the [LICENSE](/LICENSE) file for details.

## Acknowledgements

Carla GymDrive is inspired by the open-source community and contributions from researchers and developers around the world. I would like to express our gratitude to the Carla team for providing an excellent simulator for autonomous driving research.

I would like to thank the user [song-hl](https://github.com/song-hl) for helping me debug the PPO custom feature extractor for the stable-baselines3 framework.
