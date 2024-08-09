"""
Use this script to modify the value of a key in a YAML file.

Usage:
    python modify_yaml.py config.yaml new_config.yaml modify_item_key new_value
"""

from ruamel.yaml import YAML
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Modify a key in a YAML configuration file and save the result to a new file.")
    parser.add_argument('input_path', type=str, help='Path to the input YAML file')
    parser.add_argument('output_path', type=str, help='Path to save the modified YAML file')
    parser.add_argument('key', type=str, help='Key to be modified (use dot notation for nested keys)')
    parser.add_argument('new_value', type=str, help='New value for the specified key')
    return parser.parse_args()

def load_yaml(file_path):
    yaml = YAML()
    with open(file_path, 'r') as file:
        return yaml.load(file)

def save_yaml(data, file_path):
    yaml = YAML()
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

def set_nested_value(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def main():
    args = parse_args()

    # Load the input YAML file
    try:
        config = load_yaml(args.input_path)
    except FileNotFoundError:
        print(f"Error: The file {args.input_path} does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        sys.exit(1)

    # Convert new_value to the appropriate type
    try:
        new_value = YAML().load(args.new_value)
    except Exception as e:
        print(f"Error parsing new value: {e}")
        sys.exit(1)

    # Modify the specified key in the YAML data
    keys = args.key.split('.')
    try:
        set_nested_value(config, keys, new_value)
    except KeyError as e:
        print(f"Error: Key {e} not found in YAML file.")
        sys.exit(1)

    # Save the modified YAML to the output file
    try:
        save_yaml(config, args.output_path)
    except Exception as e:
        print(f"Error saving YAML file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
