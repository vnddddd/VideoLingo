import os
from ruamel.yaml import YAML
import shutil
import threading

CONFIG_PATH = 'config.yaml'
CONFIG_EXAMPLE_PATH = 'config.example.yaml'
lock = threading.Lock()

yaml = YAML()
yaml.preserve_quotes = True

# -----------------------
# load & update config
# -----------------------

def ensure_config_file():
    """Create local config.yaml from config.example.yaml when it is missing."""
    if os.path.exists(CONFIG_PATH):
        return
    if not os.path.exists(CONFIG_EXAMPLE_PATH):
        raise FileNotFoundError(
            f"{CONFIG_PATH} not found and {CONFIG_EXAMPLE_PATH} is missing"
        )
    shutil.copy2(CONFIG_EXAMPLE_PATH, CONFIG_PATH)

def load_key(key):
    with lock:
        ensure_config_file()
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

    keys = key.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise KeyError(f"Key '{k}' not found in configuration")
    return value

def update_key(key, new_value):
    with lock:
        ensure_config_file()
        with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
            data = yaml.load(file)

        keys = key.split('.')
        current = data
        for k in keys[:-1]:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return False

        if isinstance(current, dict) and keys[-1] in current:
            current[keys[-1]] = new_value
            with open(CONFIG_PATH, 'w', encoding='utf-8') as file:
                yaml.dump(data, file)
            return True
        else:
            raise KeyError(f"Key '{keys[-1]}' not found in configuration")


def load_timeout(key, default):
    """Read request_timeout.<key> as a positive seconds value, with safe fallback."""
    try:
        value = load_key(f'request_timeout.{key}')
    except KeyError:
        return default

    if value is None or value == '':
        return default

    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return default

    if timeout <= 0:
        return default

    return int(timeout) if timeout.is_integer() else timeout
def load_positive_int(key, fallback_key=None, default=1):
    """Read a positive integer config value, optionally falling back to another key."""
    try:
        value = load_key(key)
    except KeyError:
        if fallback_key is None:
            value = default
        else:
            try:
                value = load_key(fallback_key)
            except KeyError:
                value = default

    if value is None or value == '':
        value = default

    try:
        number = int(value)
    except (TypeError, ValueError):
        number = int(default)

    return max(1, number)
        
# basic utils
def get_joiner(language):
    if language in load_key('language_split_with_space'):
        return " "
    elif language in load_key('language_split_without_space'):
        return ""
    else:
        raise ValueError(f"Unsupported language code: {language}")

if __name__ == "__main__":
    print(load_key('language_split_with_space'))
