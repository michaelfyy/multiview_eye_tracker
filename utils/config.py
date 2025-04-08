import yaml
import argparse

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Pupil and Gaze Training")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to the config file')
    args = parser.parse_args()
    config = load_config(args.config)
    return config
