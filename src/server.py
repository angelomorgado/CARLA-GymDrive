import os
import subprocess
import time

'''
Server Module

This module contains the CarlaServer class that is responsible for starting and stopping the Carla server.

Requirements:
    - Environment variable CARLA_SERVER that contains the path to the Carla server directory
'''

class CarlaServer:
    @staticmethod
    def initialize_server(low_quality = False, offscreen_rendering = False, silent = False, sleep_time = 10):
        # Get environment variable CARLA_SERVER that contains the path to the Carla server directory
        carla_server = os.getenv('CARLA_SERVER')

        # If it is Unix add the CarlaUE4.sh to the path else add CarlaUE4.exe
        if os.name == 'posix':
            carla_server = os.path.join(carla_server, 'CarlaUE4.sh')
            command = f"bash {carla_server} {'--quality-level=Low' if low_quality else ''} {'--RenderOffScreen' if offscreen_rendering else ''}"
        else:
            carla_server = os.path.join(carla_server, 'CarlaUE4.exe')
            command = f"{carla_server} {'--quality-level=Low' if low_quality else ''} {'--RenderOffScreen' if offscreen_rendering else ''}"

        # Run the command
        if not silent:
            print('Starting Carla server, please wait...')
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
        # Wait for the server to start
        time.sleep(sleep_time)
        if not silent:
            print('Carla server started')

        return process
    
    @staticmethod
    def close_server(process, silent = False):
        if os.name == 'posix':
            os.killpg(os.getpgid(process.pid), 15)
            if not silent:
                print('Carla server closed')
        else:
            # On Windows, use taskkill to terminate the process and all its children
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not silent:
                print('Carla server closed')
    
    @staticmethod
    def kill_carla_linux():
        if os.name == 'posix':
            os.system('pkill -9 -f CarlaUE4')
            print('Carla server closed')
        else:
            print('This method is only for Unix systems! Please close the Carla server manually.')
