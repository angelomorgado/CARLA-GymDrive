# Pygame Window
IM_WIDTH                = 1920
IM_HEIGHT               = 1080
NUM_COLS                = 2  # Number of columns in the grid
NUM_ROWS                = 2  # Number of rows in the grid
MARGIN                  = 30
BORDER_WIDTH            = 5

# Vehicle and Sensors attributes
SENSOR_FPS              = 30
VERBOSE                 = False
VEHICLE_SENSORS_FILE    = 'env/train_sensors.json'
VEHICLE_PHYSICS_FILE    = 'env/default_vehicle_physics.json'
VEHICLE_MODEL           = "vehicle.tesla.model3"

# Simulation attributes
SIM_HOST                = 'localhost'
SIM_PORT                = 2000
SIM_TIMEOUT             = 100.0
SIM_LOW_QUALITY         = False
SIM_OFFSCREEN_RENDERING = False
SIM_DELTA_SECONDS       = 0.05
SIM_FPS                 = 30

# Environment attributes
ENV_SCENARIOS_FILE      = 'env/simple_scenarios.json'
ENV_MAX_STEPS           = 3500 # Used to limit the number of steps in the environment and to calculate the reward for finishing an episode successfully
ENV_REWARDS_LAMBDAS     = {
                            'orientation': 1,
                            'distance': -3,
                            'time_driving': 0.00005,
                            'speed': -2,
                            'destination': 3.0,
                            'collision': -10,
                            'light_pole_transgression': -10,
                            'stop_sign_transgression': -10,
                            'throttle_jerk': -2,
                            'steering_jerk': -2,
                            'time_limit': -5,
                         }
