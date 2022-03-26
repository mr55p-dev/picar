import argparse
from PyCrashed.models import NVidia, ImageNetPretrained, MultiHeaded, Model
from PyCrashed.pipeline import Dataset

from tabulate import tabulate

models = {
    "nvidia": NVidia,
    "imagenet": ImageNetPretrained,
    "multiheaded": MultiHeaded
}
model_names = ', '.join(models.keys())
parser = argparse.ArgumentParser(description='Train some models.')
parser.add_argument("model", help=f" The name of the model. Current models are: {model_names} ")
parser.add_argument("-e", "--epochs", help="Number of epochs to train for.", type=int, default=10)
parser.add_argument("-b", "--batch", help="Number of instances in a batch.", type=int)
parser.add_argument("-nt", "--n-train", help="Number of instances to use for training.", type=int)
parser.add_argument("-nv", "--n-val", help="Number of instances to use for validation.", type=int)
parser.add_argument("--no-stopping", help="Control using early stopping", default=True, action="store_true", dest="earlystopping")
parser.add_argument("--no-logging", help="Control tensorboard and remote logging", default=True, action="store_false", dest="logging")
parser.add_argument("--no-checkpoints", help="Control model checkpoints", default=True, action="store_false", dest="checkpoints")
args = parser.parse_args()

print("Configuring data pipeline")
if args.n_train:
    Dataset.set("N_TRAIN", args.n_train)
if args.batch:
    Dataset.set("N_VAL", args.n_val)
if args.n_val:
    Dataset.set("BATCH_SIZE", args.batch)


print("Instantiating model")
model: Model = models[args.model](use_logging=args.logging, use_early_stopping=args.earlystopping, use_checkpoints=args.checkpoints)

print("Building model")
model.build().summary()

fmt = lambda k: k.replace('_', ' ').title()
print("Executing model using the following dataset configuration")
print(tabulate({fmt(k): [v] for k, v in Dataset._props.items()}, headers="keys", tablefmt="fancy_grid"))

print("Training model")
model.fit(n_epochs=args.epochs)

print("Testing model")
model.test()

print("Saving model")
model.save()
