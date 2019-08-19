import os
import yaml
import easydict
from os.path import join


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


import argparse
parser = argparse.ArgumentParser(description='Code for *Learning to Transfer Examples for Partial Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()

config_file = args.config

args = yaml.load(open(config_file))

save_config = yaml.load(open(config_file))

args = easydict.EasyDict(args)

dataset = None

import torchvision.datasets as ds
if args.data.dataset.name == 'digital':
    dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['mnist', 'usps'],
        files=[
            ds.MNIST,
            ds.USPS
        ],
        prefix=args.data.dataset.root_path)

elif args.data.dataset.name == 'office':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['amazon', 'dslr', 'webcam'],
    files=[
        'amazon_reorgnized.txt',
        'dslr_reorgnized.txt',
        'webcam_reorgnized.txt'
    ],
    prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'officehome':
    dataset = Dataset(
    path=args.data.dataset.root_path,
    domains=['Art', 'Clipart', 'Product', 'Real_World'],
    files=[
        'Art.txt',
        'Clipart.txt',
        'Product.txt',
        'Real_World.txt'
    ],
    prefix=args.data.dataset.root_path)
else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')

source_domain_name = dataset.domains[args.data.dataset.source]
target_domain_name = dataset.domains[args.data.dataset.target]
source_file = dataset.files[args.data.dataset.source]
target_file = dataset.files[args.data.dataset.target]
