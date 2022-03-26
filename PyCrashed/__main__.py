import argparse
from PyCrashed.models import NVidia, ImageNetPretrained, MultiHeaded, Model

models = {
    "nvidia": NVidia,
    "imagenet": ImageNetPretrained,
    "multiheaded": MultiHeaded
}

parser = argparse.ArgumentParser(description='Train some models.')
parser.add_argument("model", help="The name of the model.")
parser.add_argument("-e", "--epochs", help="Number of epochs to train for.", type=int, default=10)
parser.add_argument("-b", "--batch", help="Number of instances in a batch.", type=int)
parser.add_argument("--no-stopping", help="Control using early stopping", default=True, action="store_true", dest="earlystopping")
parser.add_argument("--no-logging", help="Control tensorboard and remote logging", default=True, action="store_false", dest="logging")
parser.add_argument("--no-checkpoints", help="Control model checkpoints", default=True, action="store_false", dest="checkpoints")
args = parser.parse_args()

print("Instantiating model")
model: Model = models[args.model](use_logging=args.logging, use_early_stopping=args.earlystopping, use_checkpoints=args.checkpoints)

print("Building model")
model.build().summary()

print("Training model")
model.fit(n_epochs=args.epochs)

print("Testing model")
model.test()

print("Saving model")
model.save()
