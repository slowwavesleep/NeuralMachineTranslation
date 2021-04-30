import argparse

from src.utils.data import split_data


parser = argparse.ArgumentParser(description='Convert raw data to test, dev, test files.')

parser.add_argument(dest='source_path', type=str, help='Path to source sentences.')
parser.add_argument(dest='target_path', type=str, help='Path to target sentences.')
parser.add_argument(dest='destination', type=str, help='Path to directory for writing results.')
parser.add_argument(dest='seed', type=int)

args = parser.parse_args()

with open(args.source_path) as file:
    source = file.readlines()

with open(args.target_path) as file:
    target = file.readlines()

assert len(source) == len(target)

split_data(source, target, args.destination, args.seed)
