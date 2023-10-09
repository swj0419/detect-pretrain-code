import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--target_model', type=str, default="text-davinci-003", help="the model to attack: huggyllama/llama-65b, text-davinci-003")
        self.parser.add_argument('--ref_model', type=str, default="huggyllama/llama-7b")
        self.parser.add_argument('--output_dir', type=str, default="out")
        self.parser.add_argument('--data', type=str, default="swj0419/WikiMIA", help="the dataset to evaluate: default is WikiMIA")
        self.parser.add_argument('--length', type=int, default=64, help="the length of the input text to evaluate. Choose from 32, 64, 128, 256")
        self.parser.add_argument('--key_name', type=str, default="input", help="the key name corresponding to the input text. Selecting from: input, parapgrase")




