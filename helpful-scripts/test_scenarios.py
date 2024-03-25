import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from env.environment import CarlaEnv

# Used to checkout all the scenarios
def env_test():
    env = CarlaEnv('carla-rl-gym_cont', initialize_server=True, has_traffic=True, verbose=True)
    active_s = 0

    while True:
        try:
            env.print_all_scenarios()
            i = int(input("Enter scenario index (-1 to exit): "))
                
            if i == -1:
                env.close()
                break

            active_s = i
            env.clean_scenario()
            env.load_scenario(env.situations_list[i])
            
            # Spectator Debugging
            env.place_spectator_above_vehicle()
            
            # Waypoint Debugging
            # env.output_waypoints(d=5.0)
            env.output_waypoints_to_target()

        except KeyboardInterrupt:
            env.close()
            break

if __name__ == '__main__':
    env_test()