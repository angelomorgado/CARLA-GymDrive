'''
This script allows you to test all the chosen scenarios, and it also outputs the path the ego vehicle should take to reach the target.
'''
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.env.environment import CarlaEnv

# Used to checkout all the scenarios
def env_test():
    env = CarlaEnv('carla-rl-gym_cont', initialize_server=False, has_traffic=False, verbose=True, show_sensor_data=False)
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
            vehicle_loc = env.get_vehicle().get_location()
            # Waypoint Debugging
            # env.output_waypoints_to_target()
            waypoints = env.get_path_waypoints(spacing=7.0)
            env.draw_waypoints(waypoints)
            print("Distance to next waypoint: ", distance(waypoints[0], vehicle_loc))

        except KeyboardInterrupt:
            env.close()
            break
        
# Function that calculates the distance between two Location objects
def distance(loc1, loc2):
    return ((loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2 + (loc1.z - loc2.z)**2)**0.5

if __name__ == '__main__':
    env_test()