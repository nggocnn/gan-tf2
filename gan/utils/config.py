import yaml
from easydict import EasyDict


def read_config(problem_type):
    with open('config/{}.yml'.format(problem_type.lower())) as f:
        config = EasyDict(yaml.load(f))
        return config
    