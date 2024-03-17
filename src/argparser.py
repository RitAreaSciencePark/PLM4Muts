import argparse
import yaml
import os

def argparser_trainer():
    parser = argparse.ArgumentParser(
                    prog='PLMfinetuning',
                    description='Finetuning of PLM for protein stability upon single point mutation',
                    #epilog='',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter) 

    parser.add_argument("FILE", type=str, help="Either Configuration file or Dataset directory")
    parser.add_argument("--output_dir",    default=os.getcwd(),                  type=str,   help="Output dir")
    parser.add_argument("--model", choices=["MSA_Finetuning", "ProstT5_Finetuning", "ESM2_Finetuning", "MSA_Baseline"],        default='MSA_Finetuning', type=str, help="Model name")
    parser.add_argument("--learning_rate", default='5e-6',     type=float, help="Learning rate")
    parser.add_argument("--max_epochs",    default='3',        type=int,   help="Max number of epochs")
    parser.add_argument("--loss_fn",       choices=['L1', 'MSE'],       default='L1',       type=str,   help="Loss function")
    parser.add_argument("--seeds",         default=[10,11,12], type=list,  help="List of three integers for random seeds")
    parser.add_argument("--max_length",    default='1024',     type=int,   help="Max length of the aminoacid sequence")
    parser.add_argument("--optimizer",     choices=['Adam', 'AdamW', 'SGD'], default='AdamW',type=str,help="Optimizer")
    parser.add_argument("--weight_decay",  default='0.01',     type=float, help="Weight Decay for AdamW and SGD")
    parser.add_argument("--momentum",      default='0.',       type=float, help="Momentum for SGD")
    parser.add_argument("--max_tokens",    default='15000',    type=int,   help="Max number of tokens for MSA")
    args = parser.parse_args()
    return args

def argparser_tester():
    parser = argparse.ArgumentParser(
                    prog='PLMtesting',
                    description='Inference of PLM for protein stability upon single point mutation. '
                    'Arguments can be passed either by configuration file (--config_file config_file_name) or by command line.',
                    #epilog='',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file", type=str,help="Configuration file")
    parser.add_argument("--snapshot_file", default=os.getcwd() + "/snapshots/MSA_Finetuning.pt", type=str, help="Snapshot file")
    parser.add_argument("--model",choices=["MSA_Finetuning", "ProstT5_Finetuning", "ESM2_Finetuning", "MSA_Baseline"],        default='MSA_Finetuning', type=str, help="Model name")
    parser.add_argument("--max_tokens",    default='15000',    type=int,   help="Max number of tokens for MSA")
    parser.add_argument("--max_length",    default='1024',     type=int,   help="Max length of the aminoacid sequence")
    parser.add_argument("--output_dir",    default=os.getcwd(),         type=str,help="Output dir")
    parser.add_argument("--dataset_dir",   default='datasets/S1465',    type=str,help="Dataset dir")
    args = parser.parse_args()
    return args

def argparser_translator():
    parser = argparse.ArgumentParser(
                    prog='ProstT5TranslationDDP',
                    description='Distributed translation using the ProstT5 model from AA to 3Di.',
                    #epilog='',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_file",  type=str,  help="Input file")
    parser.add_argument("--output_file", default=os.getcwd() + "/translation/td.csv", type=str, help="Output file")
    parser.add_argument("--seeds",       default=[10,11,12], type=list,  help="List of three integers for random seeds")
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



