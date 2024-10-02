# CARLA GymDrive

Carla GymDrive is a powerful framework designed to facilitate reinforcement learning experiments in autonomous driving using the Carla simulator. By providing a gymnasium-like environment, it offers an intuitive and efficient platform for training driving agents using reinforcement learning techniques.

![CARLA-GymDrive](./gifs/clip.gif)

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

**WARNING:** In order to use this framework you should first download the CARLA simulator in your machine in the following [link](https://github.com/carla-simulator/carla/releases)

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

0. In order to use the software, the user first needs to install the CARLA simulator, which is the [server](https://github.com/carla-simulator/carla/releases/tag/0.9.15). After making sure that the server works, the user can then follow the installation guide in its documentation in order to use the framework. For first time users of the CARLA simulator, its installation can sometimes be tricky due to its engine, sometimes requiring drivers. Therefore, it is important to make sure that the simulator works before using the framework.

1. It is recommended to use a virtual environment with python 3.8.

- Create a new environment with `conda create -n carla python=3.8`, then open up the environment with `conda activate carla`, and then install the requirements with `pip install -r requirements.txt`

2. Set the environment variable `CARLA_SERVER` to the path of the Carla server directory:
  - On Windows:
  Open the Command Prompt or PowerShell and run:
  ```setx CARLA_SERVER "C:\path\to\Carla\server"```

  If you wish to do it through the control panel follow [this guide](https://superuser.com/questions/949560/how-do-i-set-system-environment-variables-in-windows-10)

  - On Linux:
  Open the terminal and add the following line to your .bashrc or .zshrc file:
  export CARLA_SERVER="/path/to/Carla/server"
  Then, run source ~/.bashrc or source ~/.zshrc to apply the changes.

3. Run the Carla server:

- If the script automatically starts the server, you can skip this step. Make sure to set the environment variable `CARLA_SERVER` to the path of the Carla server directory.
- If the script does not automatically start the server, you need to start the server manually.

4. Run training/testing scripts.

- Try out the framework by running `python main.py`. If you want to check out the ego vehicle moving around the map, set the autopilot variable to True in the gym.make() function in the main.py file. You should be seeing the ego vehicle moving around the map. 

---

## Environment Configuration

There are countless options for configuring the simulation and the gym environment. In order to fine-tune the environment to your needs, you can change the following parameters in the [`src/config/configuration.py`](`src/config/configuration.py`) file.

---

## Known Issues

- The simulator may crash after a certain amount of episodes. This is a known issue with Carla and is not a problem with the template. The problem happens because the CARLA server runs out of memory. This issue is reported [here](https://github.com/carla-simulator/carla/issues/3197). However, i've implemented a workaround that reloads the map every n episodes, this is the `self.__restart_every` variable in the `CarlaEnv` class. This is not a definitive solution, as it requires the CARLA devs to fix the root of the problem, but it helps to mitigate it;
- If the simulator is ran in low quality mode, it crashes the program, this is a problem in Carla's side and it's known by the community. Issue reported [here](https://github.com/carla-simulator/carla/issues/6399);
- Moving the walkers causes segmentation fault. This is a known problem with the simulator between the community [here](https://github.com/carla-simulator/carla/issues/4155);
- If you have `Out of video memory trying to allocate a rendering resource` error please run the simulator with dx11, as such: `./CarlaUE4.sh -dx11` or `CarlaUE4.exe -dx11`.

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

I would like to thank the user [song-hl](https://github.com/song-hl) for helping me debug the PPO custom feature extractor for the stable-baselines3 framework, and the user [the-big-bad-wolf](https://github.com/the-big-bad-wolf) for noting out the no rendering not updating every episode.
