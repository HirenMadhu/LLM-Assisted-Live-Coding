import os
import time
from psonic import *
from pythonosc import udp_client

# Initialize Sonic Pi before running the scipt
# Use following commands to connect to running server
localhost = "127.0.0.1"
spider_log = os.path.expanduser("~/.sonic-pi/log/spider.log")
set_server_parameter_from_log(localhost)
set_server_parameter_from_log(localhost, spider_log)


def get_token_and_server_port():
    with open(spider_log, "r") as file:
        log_content = file.read()
        token_match = re.search(r"Token: ([-]?\d+)", log_content)
        server_port_match = re.search(r"server_port=>(\d+)", log_content)
        if token_match and server_port_match:
            token = int(token_match.group(1))
            server_port = int(server_port_match.group(1))
            return token, server_port
        else:
            raise ValueError("Token or GUI port not found in the log file.")


def stop_running(ip, port, token):
    try:
        client = udp_client.SimpleUDPClient(ip, port)

        osc_address = "/stop-all-jobs"
        # Arguments for python-osc need to be in a list or tuple
        client.send_message(osc_address, token)

    except Exception as e:
        print(f"Error sending OSC message: {e}")


def read_code(dir, filename):
    with open(os.path.join(dir, filename), "r") as file:
        sonic_pi_code = file.read()
    return sonic_pi_code


def play_record(sonic_pi_code, recording_duration, recording_filename):
    # Run the Sonic Pi code
    run(sonic_pi_code)
    # Sleep to allow the code to start
    time.sleep(3)

    # Start recording
    print("Starting recording...")
    start_recording()
    # Sleep for the duration of the recording
    time.sleep(recording_duration)
    # Stop recording
    stop_recording()
    print("Stopped recording...")
    # Wait to ensure the recording is saved
    time.sleep(1)

    # Save recording to file
    save_recording(recording_filename)
    print(f"Recording saved to {recording_filename}")


def create_recording(dir_code, file_code, recording_filename):
    token, server_port = get_token_and_server_port()
    # Recording Parameters
    recording_duration = 8  # seconds (saves +2 seconds)
    # Read the Sonic Pi code from the file
    sonic_pi_code = read_code(dir_code, file_code)

    # Play and record code
    play_record(sonic_pi_code, recording_duration, recording_filename)

    # Stop the Sonic Pi code before starting next
    stop_running(localhost, server_port, token)
    # Sleep to ensure the code has stopped
    time.sleep(2)


if __name__ == "__main__":
    cwd = os.getcwd()
    dataset_dir = "../datasets/"
    recordings_dir = "../recordings"
    token, server_port = get_token_and_server_port()

    # Recording Parameters
    recording_duration = 18  # seconds (saves +2 seconds)

    # Iterate through all directories in the dataset directory
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith(".pi"):

                # remove .pi from filename
                filename = filename.split(".pi")[0]
                # Check if recording already exists for this file
                # recording filenames are under recordings/dirname/filename.wav
                dir_name = os.path.basename(dirpath)

                recordings_path = os.path.join(cwd, recordings_dir, dir_name)

                # Check if directory exists in recording_dir
                if not os.path.exists(recordings_path):
                    os.makedirs(recordings_path)

                recording_filename = os.path.join(
                    cwd, recordings_path, f"{filename}.wav"
                )
                if os.path.exists(recording_filename):
                    print(f"Recording for {filename} already exists. Skipping...")
                    continue

                # Read the Sonic Pi code from the file
                sonic_pi_code = read_code(dirpath, filename)

                # Play and record code
                play_record(sonic_pi_code, recording_filename)

                # Stop the Sonic Pi code before starting next
                stop_running(localhost, server_port, token)
                # Sleep to ensure the code has stopped
                time.sleep(3)
