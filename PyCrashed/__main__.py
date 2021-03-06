import argparse
from PyCrashed.utils import convert, list_models, train_model, predict

parser = argparse.ArgumentParser(description='Train some models.')
subparsers = parser.add_subparsers(help='Available sub-commmands')

# Subcommand to display all the available models
list_command = subparsers.add_parser('list', help='Displays the currently available models')
list_command.add_argument("--model", help="Display a model summary", type=str)
list_command.set_defaults(func=list_models)

# Subcommand to train a model
train_command = subparsers.add_parser('train', help='Train a model')
train_command.add_argument(
    "model", 
    help=" The name of the model."
)
train_command.add_argument(
    "-e", "--epochs", 
    help="Number of epochs to train for.", 
    type=int, default=20
)
train_command.add_argument(
    "-b", "--batch", 
    help="Number of instances in a batch.", 
    type=int, default=64
)
train_command.add_argument(
    "-nt", "--train", 
    help="Number of instances to use for training. (percentage)", 
    type=float, default=.75
)
train_command.add_argument(
    "-nv", "--val", 
    help="Number of instances to use for validation. (percentage)", 
    type=float, default=.25
)
train_command.add_argument(
    "--restore", 
    help="Restore the last saved checkpoint for this model", 
    default=False, action="store_true"
)
train_command.add_argument(
    "-a", "--activation", 
    help="Activation funciton (default is relu)", 
    type=str, default=None,
)
train_command.add_argument(
    "-o", "--optimizer", 
    help="Optimizer to use (default is Adam)", 
    type=str, default=None 
)
train_command.add_argument(
    "-l", "--loss", 
    help="Loss function to use (default is MSE)", 
    type=str, default=None, 
)
train_command.add_argument(
    "-d", "--dropout", 
    help="Drouput rate to use in the NN", 
    type=float, default=0.0, 
)
train_command.add_argument(
    "-p", "--paitence", 
    help="How much paitence to have", 
    type=int, default=None
)
train_command.add_argument(
    "-q", "--silent", 
    help="Hide model pregress", default=True, 
    action="store_false", dest="verbose"
)
train_command.set_defaults(func=train_model)

# Subcommand to generate a predictions csv file 
predict_command = subparsers.add_parser('predict', help='Make predictions on the unseen data')
predict_command.add_argument("path", help="The path to a saved model file")
predict_command.add_argument("-o", "--output", help="The output file destination")
predict_command.add_argument("--silent", help="Hide pregress", default=True, action="store_false", dest="verbose")
predict_command.set_defaults(func=predict)

# Subcommand to convert a saved model into a tflite binary
convert_command = subparsers.add_parser('convert', help='Convert a model to tflite')
convert_command.add_argument("path", help="The path to a saved model file")
convert_command.add_argument("-o", "--output", help="The output file destination")
convert_command.set_defaults(func=convert)

args = parser.parse_args()
args.func(args)
