import argparse
from PyCrashed.utils import train_model

parser = argparse.ArgumentParser(description='Train some models.')

parser.add_argument(
    "--model", help=" The name of the model.",
    type=str, default=None
)
parser.add_argument(
    "-e", "--epochs", 
    help="Number of epochs to train for.", 
    type=int, default=20
)
parser.add_argument(
    "-b", "--batch", 
    help="Number of instances in a batch.", 
    type=int, default=64
)
parser.add_argument(
    "-nt", "--train", 
    help="Number of instances to use for training. (percentage)", 
    type=float, default=.65
)
parser.add_argument(
    "-nv", "--val", 
    help="Number of instances to use for validation. (percentage)", 
    type=float, default=.25
)
parser.add_argument(
    "--restore", 
    help="Restore the last saved checkpoint for this model", 
    default=False, action="store_true"
)
parser.add_argument(
    "-a", "--activation", 
    help="Activation funciton (default is relu)", 
    type=str, default=None, 
)
parser.add_argument(
    "-o", "--optimizer", 
    help="Optimizer to use (default is Adam)", 
    type=str, default=None 
)
parser.add_argument(
    "-l", "--loss", 
    help="Loss function to use (default is MSE)", 
    type=str, default=None, 
)
parser.add_argument(
    "-d", "--dropout", 
    help="Drouput rate to use in the NN", 
    type=float, default=0.0, 
)
parser.add_argument(
    "-kw", "--kernel-width", 
    help="Multiplier for the number of kernels used", 
    type=float, default=1, 
)
parser.add_argument(
    "-nw", "--network-width", 
    help="Multiplier for the number of nodes in the neural network", 
    type=float, default=1, 
)
parser.add_argument(
    "-p", "--paitence", 
    help="How much paitence to have", 
    type=int, default=None
)
parser.add_argument(
    "-q", "--silent", 
    help="Hide model pregress", default=True, 
    action="store_false", dest="verbose"
)
parser.set_defaults(func=train_model)
args = parser.parse_args()
args.func(args)