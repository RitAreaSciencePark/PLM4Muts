import argparse
import yaml
import os

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default='ProstT5_Milano',    type=str,   help="Model name")
    parser.add_argument("--lr",          default='1e-5',      type=float, help="Learning rate")
    parser.add_argument("--optimizer",   default='Adam',      type=str,   help="Optimizer")
    parser.add_argument("--max_epochs",  default='3',         type=int,   help="Number of epochs")
    parser.add_argument("--output_dir",  default=os.getcwd(), type=str,   help="Output dir path.")
    parser.add_argument("--config_file", default=os.getcwd()+"/config.yaml", type=str, help="Output dir")
    parser.add_argument("--dataset_dir", default='datasets/standard',type=str,help="Dataset")
    parser.add_argument("--loss_fn",   default='MSE',type=str,help="Loss function. Choose 'MSE' or 'L1'.")
    parser.add_argument("--device",    default='cuda', type=str, help="Device: choose 'cpu' or 'cuda'.")
    parser.add_argument("--verbose",   default=True, type=eval)
    parser.add_argument("--save_every",default='2', type=int, help="???")
    args = parser.parse_args()
    return args


def argparser_translator():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",  default='datasets/standard/train/MSA_databases/db_train.csv', type=str,  help="In")
    parser.add_argument("--output_file", default='datasets/standard/train/MSA_3Di_databases/tb_train.csv', type=str, help="Out")
    args = parser.parse_args()
    return args

# Function to load yaml configuration file
def load_config(confir_file):
    with open(confir_file) as file:
        config = yaml.safe_load(file)

    return config

