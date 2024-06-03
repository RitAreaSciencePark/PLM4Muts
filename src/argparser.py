import argparse
import yaml
import os

def argparser_trainer():
    parser = argparse.ArgumentParser(
                    prog='PLMfinetuning',
                    description='Finetuning of PLM for protein stability upon single point mutation',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter) 
    parser.add_argument("config_file", type=str, help="Configuration file")
    args = parser.parse_args()
    return args

def argparser_tester():
    parser = argparse.ArgumentParser(
                    prog='PLMtesting',
                    description='Inference of PLM for protein stability upon single point mutation.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", type=str, help="Configuration file")
    args = parser.parse_args()
    return args

def argparser_translator():
    parser = argparse.ArgumentParser(
                    prog='ProstT5TranslationDDP',
                    description='Distributed translation using the ProstT5 model from AA to 3Di.',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_file",  type=str,  help="Input file")
    parser.add_argument("--output_file", default=os.getcwd() + "/translation/td.csv", type=str, help="Output file")
    parser.add_argument("--max_length",  default='490',     type=int,   help="Max length of the aminoacid sequence")
    parser.add_argument("--seeds",       default=[10,11,12], type=list,  help="List of three integers for random seeds")
    args = parser.parse_args()
    return args


def argparser_onnx_inspector():
    parser = argparse.ArgumentParser(
                    prog='onnx inspector',
                    description='Check the onnx model parameter',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config_file", type=str, help="Configuration file")
    args = parser.parse_args()
    return args

# Function to load yaml configuration file
def load_config(config_file):
    with open(config_file, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as error:
            print(error)
            raise SystemExit(1)
    return config



