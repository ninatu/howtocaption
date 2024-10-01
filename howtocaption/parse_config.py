import os
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
import time
import inspect
from collections import OrderedDict
import json

from howtocaption.utils import read_yaml, write_yaml


def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


class ConfigParser:
    def __init__(self, args_parser, options='', timestamp=True, test=False, parse_from_string=None):
        # parse default and custom cli options
        for opt in options:
            args_parser.add_argument(*opt.flags, default=None, type=opt.type)

        if parse_from_string is not None:
            import shlex
            args = args_parser.parse_args(shlex.split(parse_from_string))
        else:
            args = args_parser.parse_args()
        self.args = args

        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg
        config = read_yaml(Path(args.config))

        if hasattr(args, 'resume') and args.resume is not None:
            self.resume = Path(args.resume)
        else:
            self.resume = None

        # load config file and apply custom cli options
        self.config = _update_config(config, options, args)

        # set seed
        self.config['seed'] = self.config.get('seed', None)  # set None if it's not given
        if hasattr(args, 'seed') and args.seed is not None:
            self.config['seed'] = args.seed

        config['name'] = os.path.splitext(os.path.basename(args.config))[0]

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['save_dir'])
        timestamp = datetime.now().strftime(r'%y%m%d_%H%M%S') if timestamp else ''

        exper_name = self.config['name']
        self._save_dir = save_dir / 'models' / exper_name / timestamp

        if not test:
            make_dir=True
            counter= 0
            while make_dir:
                try:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    make_dir = False
                except PermissionError:
                    exper_name = f'{self.config["name"]}_{counter}'
                    self._save_dir = save_dir / 'models' / exper_name / timestamp
                    self.config['name'] = exper_name
                    counter += 1

        # save updated config file to the checkpoint dir
        if not test:
            write_yaml(self.config, self.save_dir / 'config.yaml')

    def initialize(self, name, module,  *args, index=None, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        if index is None:
            module_name = self[name]['type']
            module_args = dict(self[name]['args'])
        else:
            module_name = self[name][index]['type']
            module_args = dict(self[name][index]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        # if parameter not in config subdict, then check if it's in global config.
        signature = inspect.signature(getattr(module, module_name).__init__)  # (self, arg1, arg2, ...)
        for param in list(signature.parameters.keys())[1 + len(args):]:  # self + first n that takes from args
            if param not in module_args and param in self.config:
                module_args[param] = self[param]
                print(f'Warning: Using param {param} from global config in {name}')

        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    @property
    def save_dir(self):
        return self._save_dir


# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
