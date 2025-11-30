"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        if arg.startswith('--'):
            # Treat bare --flag as boolean True override if it exists and is bool
            key = arg[2:]
            if key in globals() and isinstance(globals()[key], bool):
                print(f"Overriding: {key} = True")
                globals()[key] = True
            else:
                # If it's not a known bool, raise to avoid silent mistakes
                raise ValueError(f"Unknown or non-bool flag used without value: {arg}. Use --{key}=<value> format.")
        else:
            # assume it's the name of a config file
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print(f.read())
            exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            # allow None->typed transitions (e.g., None to str) by relaxing strict type match
            if globals()[key] is not None and type(attempt) != type(globals()[key]):
                raise TypeError(f"Type mismatch for {key}: have {type(globals()[key]).__name__}, new {type(attempt).__name__}")
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
