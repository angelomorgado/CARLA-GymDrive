'''
Display Module:
    It provides the functionality to display the sensor data in a window using Pygame.
'''
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

import configuration

class Display:
    def __init__(self, title, vehicle):
        self.__sensor_window_dict = {}
        self.__sensor_dict = vehicle.get_sensor_dict()
        self.__non_displayable_sensors = ['gnss', 'imu', 'collision', 'lane_invasion']
        self.__main_screen = self.__initialize_pygame_window(title)
        self.__clock = pygame.time.Clock()  # Create a clock object to control the frame rate

    def __initialize_pygame_window(self, title):
        pygame.init()
        pygame.display.set_caption(title)

        # Initialize the sensor windows
        for sensor in self.__sensor_dict:
            if sensor not in self.__non_displayable_sensors:
                self.__sensor_window_dict[sensor] = pygame.Surface((640, 360))

        return pygame.display.set_mode((configuration.IM_WIDTH, configuration.IM_HEIGHT))

    # This play_window function is used to display the sensor data in a window using Pygame. However, it has its own event loop, which is not suitable for the main loop of the program since it can't be ran in a thread.
    def play_window(self):
        clock = pygame.time.Clock()  # Create a clock object to control the frame rate

        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                self.__main_screen.fill((127, 127, 127))  # Fill the main window with a gray background

                for idx, sensor in enumerate(self.__sensor_window_dict):
                    sub_surface = self.__sensor_window_dict[sensor]
                    sub_surface_width, sub_surface_height = sub_surface.get_size()

                    # Calculate row and column index
                    row_idx = idx // configuration.NUM_COLS
                    col_idx = idx % configuration.NUM_COLS

                    x_position = configuration.MARGIN + col_idx * (sub_surface_width + configuration.MARGIN)
                    y_position = configuration.MARGIN + row_idx * (sub_surface_height + configuration.MARGIN)

                    # Draw a border around each sub-surface
                    pygame.draw.rect(self.__main_screen, (50, 50, 50), (x_position - configuration.BORDER_WIDTH, y_position - configuration.BORDER_WIDTH,
                                                                sub_surface_width + 2 * configuration.BORDER_WIDTH,
                                                                sub_surface_height + 2 * configuration.BORDER_WIDTH), configuration.BORDER_WIDTH)

                    # Display each sub-surface with a margin
                    self.__main_screen.blit(sub_surface, (x_position, y_position))

                    # Check if the active_img is not None before blitting it
                    if sensor in self.__sensor_dict and self.__sensor_dict[sensor].get_last_data() is not None:
                        pygame_surface = pygame.surfarray.make_surface(self.__sensor_dict[sensor].get_last_data().swapaxes(0, 1))
                        self.__main_screen.blit(pygame_surface, (x_position, y_position))

                    # Display sensor legend
                    font = pygame.font.Font(None, 24)
                    legend_text = font.render(sensor.capitalize(), True, (255, 255, 255))
                    self.__main_screen.blit(legend_text, (x_position + 10, y_position + sub_surface_height - 30))

                # Display GNSS data
                if 'gnss' in self.__sensor_dict and self.__sensor_dict['gnss'].get_last_data() is not None: 
                    gnss_font = pygame.font.Font(None, 24)
                    gnss_data = self.__sensor_dict['gnss'].get_last_data()
                    gnss_text = f"GNSS Sensor: Latitude {gnss_data.latitude:.6f}, Longitude {gnss_data.longitude:.6f}, Altitude {gnss_data.altitude:.6f}"
                    gnss_surface = gnss_font.render(gnss_text, True, (255, 255, 255))
                    self.__main_screen.blit(gnss_surface, (configuration.MARGIN, configuration.IM_HEIGHT - configuration.MARGIN))

                # Display IMU data
                if 'imu' in self.__sensor_dict and self.__sensor_dict['imu'].get_last_data() is not None:
                    imu_font = pygame.font.Font(None, 24)
                    imu_data = self.__sensor_dict['imu'].get_last_data()
                    imu_text = f"IMU Sensor: Acceleration {imu_data.accelerometer.x:.6f}, {imu_data.accelerometer.y:.6f}, {imu_data.accelerometer.z:.6f}," \
                            f"Gyroscope {imu_data.gyroscope.x:.6f}, {imu_data.gyroscope.y:.6f}, {imu_data.gyroscope.z:.6f}, " \
                            f"Compass {imu_data.compass:.6f}"
                    imu_surface = imu_font.render(imu_text, True, (255, 255, 255))
                    imu_text_rect = imu_surface.get_rect()
                    imu_text_rect.topleft = (configuration.IM_WIDTH - imu_text_rect.width - configuration.MARGIN, configuration.IM_HEIGHT - configuration.MARGIN)
                    self.__main_screen.blit(imu_surface, imu_text_rect)

                pygame.display.flip()

                # Limit the frame rate to SENSOR_FPS
                clock.tick(configuration.SENSOR_FPS)

        finally:
            pygame.quit()
            print('Display window closed!')
    
    # This play_window funciton does not have its own event loop, therefore it needs to be called inside the main loop of the program.
    def play_window_tick(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.__main_screen.fill((127, 127, 127))  # Fill the main window with a gray background

        for idx, sensor in enumerate(self.__sensor_window_dict):
            sub_surface = self.__sensor_window_dict[sensor]
            sub_surface_width, sub_surface_height = sub_surface.get_size()

            # Calculate row and column index
            row_idx = idx // configuration.NUM_COLS
            col_idx = idx % configuration.NUM_COLS

            x_position = configuration.MARGIN + col_idx * (sub_surface_width + configuration.MARGIN)
            y_position = configuration.MARGIN + row_idx * (sub_surface_height + configuration.MARGIN)

            # Draw a border around each sub-surface
            pygame.draw.rect(self.__main_screen, (50, 50, 50), (x_position - configuration.BORDER_WIDTH, y_position - configuration.BORDER_WIDTH,
                                                        sub_surface_width + 2 * configuration.BORDER_WIDTH,
                                                        sub_surface_height + 2 * configuration.BORDER_WIDTH), configuration.BORDER_WIDTH)

            # Display each sub-surface with a margin
            self.__main_screen.blit(sub_surface, (x_position, y_position))

            # Check if the active_img is not None before blitting it
            if sensor in self.__sensor_dict and self.__sensor_dict[sensor].get_last_data() is not None:
                pygame_surface = pygame.surfarray.make_surface(self.__sensor_dict[sensor].get_last_data().swapaxes(0, 1))
                self.__main_screen.blit(pygame_surface, (x_position, y_position))

            # Display sensor legend
            font = pygame.font.Font(None, 24)
            legend_text = font.render(sensor.capitalize(), True, (255, 255, 255))
            self.__main_screen.blit(legend_text, (x_position + 10, y_position + sub_surface_height - 30))

        # Display GNSS data
        if 'gnss' in self.__sensor_dict and self.__sensor_dict['gnss'].get_last_data() is not None: 
            gnss_font = pygame.font.Font(None, 24)
            gnss_data = self.__sensor_dict['gnss'].get_last_data()
            gnss_text = f"GNSS Sensor: Latitude {gnss_data.latitude:.6f}, Longitude {gnss_data.longitude:.6f}, Altitude {gnss_data.altitude:.6f}"
            gnss_surface = gnss_font.render(gnss_text, True, (255, 255, 255))
            self.__main_screen.blit(gnss_surface, (configuration.MARGIN, configuration.IM_HEIGHT - configuration.MARGIN))

        # Display IMU data
        if 'imu' in self.__sensor_dict and self.__sensor_dict['imu'].get_last_data() is not None:
            imu_font = pygame.font.Font(None, 24)
            imu_data = self.__sensor_dict['imu'].get_last_data()
            imu_text = f"IMU Sensor: Acceleration {imu_data.accelerometer.x:.6f}, {imu_data.accelerometer.y:.6f}, {imu_data.accelerometer.z:.6f}," \
                    f"Gyroscope {imu_data.gyroscope.x:.6f}, {imu_data.gyroscope.y:.6f}, {imu_data.gyroscope.z:.6f}, " \
                    f"Compass {imu_data.compass:.6f}"
            imu_surface = imu_font.render(imu_text, True, (255, 255, 255))
            imu_text_rect = imu_surface.get_rect()
            imu_text_rect.topleft = (configuration.IM_WIDTH - imu_text_rect.width - configuration.MARGIN, configuration.IM_HEIGHT - configuration.MARGIN)
            self.__main_screen.blit(imu_surface, imu_text_rect)

        pygame.display.flip()

        self.__clock.tick(configuration.SENSOR_FPS)
    
    def close_window(self):
        pygame.quit()
        print('Display window closed!')