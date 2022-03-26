# sourcery skip: swap-if-expression
import argparse
from tabnanny import verbose
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
parser.add_argument("--predict", help="Should the model make predictions on the unknown set", default=False, action="store_true")
group = parser.add_mutually_exclusive_group()
group.add_argument("--summary", help="Display a model summary", default=False, action="store_true")
group.add_argument("--silent", help="Hide model pregress", default=True, action="store_false", dest="verbose")
parser.add_argument("--no-train", help="Should the model be fitted to the data", default=True, action="store_false", dest="train")
parser.add_argument("--no-test", help="Should the model be tested on the data", default=True, action="store_false", dest="test")
parser.add_argument("--no-save", help="Should the model be saved", default=True, action="store_false", dest="save")
parser.add_argument("--no-stopping", help="Control using early stopping", default=True, action="store_false", dest="earlystopping")
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

printf = lambda x: print(x) if args.verbose else lambda x: None
printf("Instantiating model")
model: Model = models[args.model](
    use_logging=args.logging,
    use_early_stopping=args.earlystopping,
    use_checkpoints=args.checkpoints,
    verbose=args.verbose
)

printf("Building model")
m = model.build()
if args.summary:
    m.summary()

fmt = lambda k: k.replace('_', ' ').title()
printf("Executing model using the following dataset configuration")
printf(tabulate({fmt(k): [v] for k, v in Dataset._props.items()}, headers="keys", tablefmt="fancy_grid"))

if args.train:
    printf("Training model")
    model.fit(n_epochs=args.epochs)

if args.test and args.train:
    printf("Testing model")
    model.test()

if args.save and args.train:
    printf("Saving model")
    model.save()
