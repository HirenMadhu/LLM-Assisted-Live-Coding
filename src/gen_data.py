import json
import os
from random import choice, randint, uniform
from jinja2 import Template


def generate_template(template_path, data):
    """
    Generate a template using Jinja2.
    Args:
        template_path (str): The path to the template file.
        data (dict): The data to render the template with.
    Returns:
        str: The rendered template content.
    """
    # Load the template
    with open(template_path, 'r') as file:
        template_content = file.read()

    # Create a Jinja2 Template object
    template = Template(template_content)

    # Render the template with the data
    rendered_content = template.render(data)

    return rendered_content

def random_float_list(min_val, max_val, size, prec=2):
    """
    Generate a list of random floats.
    Args:
        min_val (float): Minimum value for the range.
        max_val (float): Maximum value for the range.
        size (int): Size of generated list.
        prec (int): Precision of the float values.
    Returns:
        list: A list of random floats.
    """
    return [round(uniform(min_val, max_val), prec) for _ in range(size)]

def random_int_list(min_val, max_val, size):
    """
    Generate a list of random integers.
    Args:
        min_val (float): Minimum value for the range.
        max_val (float): Maximum value for the range.
        size (int): Size of generated list.
    Returns:
        list: A list of random integer.
    """
    return [randint(min_val, max_val) for _ in range(size)]


def gen_values(data_dir):
    """
    Generate values for the templates using
    given data and randomizing necessary fields.
    Args:
        data_dir (str): The directory containing the data files.
    Returns:
        dict: A dictionary containing the generated values.
    """

    # For values that may appear multiple times like sleep, attack, release, amp
    # create a list of random values so entries can wrap over
    list_size = 10
    sleep_min = data_dir["sleep_range"][0]
    sleep_max = data_dir["sleep_range"][1]

    attack_min = data_dir["attack_range"][0]
    attack_max = data_dir["attack_range"][1]

    release_min = data_dir["release_range"][0]
    release_max = data_dir["release_range"][1]

    amp_min = data_dir["amp_range"][0]
    amp_max = data_dir["amp_range"][1]

    # Create lists with values that can be reused by multiple keywords
    
    # List of random floats between 0 and 1
    repeat_probs = random_float_list(0, 1, list_size)

    # List of random ints between 1 and 20 (e.g. for times loop)
    repeat_small_int = random_int_list(1, 20, list_size)

    # List of random ints between 20 and 60
    repeat_med_int = random_int_list(20, 60, list_size)

    # List of large ints (e.g. bpm, cutoff)
    repeat_large_int = random_int_list(60, 200, list_size)

    data_res = {
        "sample_values": [choice(data_dir["samples"]) for _ in range(list_size)],
        "samples_bpm": [choice(data_dir["samples_bpm"]) for _ in range(list_size)],
        "sample_names": [choice(data_dir["sample_names"]) for _ in range(list_size)],
        "sleep_values": random_float_list(sleep_min, sleep_max, list_size),
        "repeat_small_ints": repeat_small_int,
        "repeat_med_ints": repeat_med_int,
        "repeat_large_ints": repeat_large_int,
        "synth_values": [choice(data_dir["synths"]) for _ in range(list_size)],
        "character_values": [choice(data_dir["character"]) for _ in range(list_size)],
        "attack_values": random_float_list(attack_min, attack_max, list_size),
        "release_values": random_float_list(release_min, release_max, list_size),
        "amp_values": random_float_list(amp_min, amp_max, list_size),
        "effect_values": [choice(data_dir["effects"]) for _ in range(list_size)],
        "note_values": [choice(data_dir["notes"]) for _ in range(list_size)],
        "repeat_probs": repeat_probs
    }
    return data_res

if __name__ == "__main__":

    # Define number of datasets to generate (per existing template)
    num_datasets = 200

    # Define the directory where the template files are located
    cwd = os.getcwd()
    template_dir = "../templates/"
    datasets_dir = "../datasets/"

    # Load the data from the JSON file
    data_path = os.path.join(cwd, template_dir, 'data.json')
    with open(data_path, 'r') as file:
        data_dir = json.load(file)

    # Parse all existing .jinja files in the template directory
    template_files = [f for f in os.listdir(template_dir) if f.endswith('.j2')]

    for template_file in template_files:
        template_name = os.path.splitext(template_file)[0]
        for idx in range(num_datasets):
            # Generate values for the templates
            generated_values = gen_values(data_dir)

            # Generate the dataset using the generated values
            template_path = os.path.join(cwd, template_dir, template_file)

            try:
                rendered_template = generate_template(template_path, generated_values)
            except Exception as e:
                print(f"Error rendering template {template_file}: {e}")
                exit(1)

            # Save template to file
            new_template_name = f"{template_name}/{template_name}_{idx}.pi"

            # Check if directory with template_name exists
            if not os.path.exists(os.path.join(cwd, datasets_dir, template_name)):
                os.makedirs(os.path.join(cwd, datasets_dir, template_name))

            output_dir = os.path.join(cwd, datasets_dir, new_template_name)
            # skip if it exists
            if os.path.exists(output_dir):
                print(f"Skipping existing file: {output_dir}")
                continue
            with open(output_dir, 'w') as output_file:
                output_file.write(rendered_template)

            print(f"Generated data: {output_dir}")