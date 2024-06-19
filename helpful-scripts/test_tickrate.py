import carla
import time

def main():
    # Connect to the Carla server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the world
    world = client.get_world()

    # Set the synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 30.0  # 30 FPS
    world.apply_settings(settings)

    try:
        tick_count = 0
        start_time = time.time()

        # Run the simulation for 10 seconds
        while time.time() - start_time < 10:
            world.tick()
            tick_count += 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        tick_rate = tick_count / elapsed_time

        print(f"Total Ticks: {tick_count}")
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")
        print(f"Tick Rate: {tick_rate:.2f} ticks/sec")

    finally:
        # Reset the settings
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

if __name__ == '__main__':
    main()