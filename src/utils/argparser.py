import argparse


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  default='ProstT5_Padriciano', type=str, help="Model name. Choose `Padriciano` or `Roma`....")
    parser.add_argument("--lr",          default='1e-5', type=float, help="Learning rate")
    parser.add_argument("--epochs",      default='1', type=int, help="Number of epochs")
    parser.add_argument("--current_dir", default='../runs', type=str, help="Current run data path.")
    parser.add_argument("--train_dir",   default='../datasets/train', type=str, help="Train dataset path.")
    parser.add_argument("--val_dir",     default='../datasets/train', type=str, help="Val dataset path.")
    parser.add_argument("--loss_method", default='MSE', type=str, help="Loss function. Choose 'MSE' or 'L1'.")
    parser.add_argument("--device", default='cuda', type=str, help="Device: choose 'cpu' or 'cuda'.")
    parser.add_argument("--verbose", default=True, type=eval)
    args = parser.parse_args()
    return args

