# Picar model training pipeline

## Setup
There shouldn't be any unusual dependencies, the environment I am using is in `environment.yml`.

## Usage
At the command line, run `python3 -m PyCrashed -h` for a list of commands. These commands are pretty self explanatory, but for a list of valid options for each run `python3 -m PyCrashed <command> --help`. 

All the models and predictions get output to a folder `products/<model name>` unless specified otherwise.

## Specifying a new model
The `BaseModel` class wraps most of the functionality, so just inherit from that and implement the `specify_model` method which returns the inputs and output functions of a Keras functional mode.