import argparse
from distutils.command.config import config
import wandb
from PyCrashed.utils import list_models, train_model, predict

parser = argparse.ArgumentParser(description='Train some models.')
subparsers = parser.add_subparsers(help='Available sub-commmands')

list_command = subparsers.add_parser('list', help='Displays the currently available models')
list_command.add_argument("--model", help="Display a model summary", type=str)
list_command.set_defaults(func=list_models)

train_command = subparsers.add_parser('train', help='Train a model')
train_command.add_argument("model", help=" The name of the model.")
train_command.add_argument("-e", "--epochs", help="Number of epochs to train for.", type=int, default=10)
train_command.add_argument("-b", "--batch", help="Number of instances in a batch.", type=int, default=10)
train_command.add_argument("-nt", "--n-train", help="Number of instances to use for training.", type=int)
train_command.add_argument("-nv", "--n-val", help="Number of instances to use for validation.", type=int)
train_command.add_argument("--silent", help="Hide model pregress", default=True, action="store_false", dest="verbose")
train_command.add_argument("--no-train", help="Should the model be fitted to the data", default=True, action="store_false", dest="train")
train_command.add_argument("--no-test", help="Should the model be tested on the data", default=True, action="store_false", dest="test")
train_command.add_argument("--no-save", help="Should the model be saved", default=True, action="store_false", dest="save")
train_command.add_argument("--no-stopping", help="Control using early stopping", default=True, action="store_false", dest="earlystopping")
train_command.add_argument("--no-logging", help="Control tensorboard and remote logging", default=True, action="store_false", dest="logging")
train_command.add_argument("--no-checkpoints", help="Control model checkpoints", default=True, action="store_false", dest="checkpoints")
train_command.set_defaults(func=train_model)

predict_command = subparsers.add_parser('predict', help='Make predictions on the unseen data')
predict_command.add_argument("path", help="The path to a saved model file")
predict_command.add_argument("-o", "--output", help="The output file destination")
predict_command.add_argument("--silent", help="Hide pregress", default=True, action="store_false", dest="verbose")
predict_command.set_defaults(func=predict)
args = parser.parse_args()

wandb.init(project="PyCrashed", entity="mr55p", config=args)

args.func(args)