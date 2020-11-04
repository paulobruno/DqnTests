# parse command line arguments
import argparse


def build_args():

    parser = argparse.ArgumentParser(
        description="Train an agent in a given scenario. The agent can be trained from scratch or load a trained model. Be careful: if loading a previously trained agent the scenario given should be the same, if not it could break or run as usual but with a poor performance.")

    parser.add_argument(
        "config_file",
        help="CFG file with settings of the ViZDoom scenario.")
    parser.add_argument(
        "-l", "--load-model",
        action="store_true",
        help="Load a model from '--model-folder' if it is given and exists.")
    parser.add_argument(
        "-s", "--save-model",
        action="store_true",
        help="Save a model in '--model-folder' if it is given. If no '--model-folder' is given, a folder called 'temp_model' will be created.")
    parser.add_argument(
        "-m", "--model-folder",
        metavar="FOLDERNAME",
        help="Path to the folder containing the model to be saved or loaded.",
        default="temp_model")
    parser.add_argument(
        "-w", "--weight-folder",
        metavar="FOLDERNAME",
        help="Path to the folder containing the checkpoints .",
        default="temp_model")
    parser.add_argument(
        "-e", "--num-epochs",
        type=int,
        metavar="N",
        help="Num of epochs to train. [default=20]",
        default=20)
    parser.add_argument(
        "-i", "--save-model-interval",
        type=int,
        metavar="N",
        help="The model will be saved every N epochs. [default=5]",
        default=5)
    parser.add_argument(
        "-show", "--show-model",
        metavar="FILENAME",
        help="Print the model architecture on screen and save a PNG image.",
        default="")
    parser.add_argument(
        "-v", "--enable-training-view",
        action="store_true",
        help="Enable rendering of the game during agent training.")
    parser.add_argument(
        "-log", "--log-file",
        metavar="FILENAME",
        help="Path to a file to save the results.",
        default="temp_log_file.txt")

    return parser.parse_args()


def log_params(args, params):
    if args.log_file:
        log_file = open(args.log_file, "w", buffering=1)
        for k, v in params.items():
            print("{key}: {value}".format(key=k, value=v))
        print("Map,{}".format(args.config_file), file=log_file)
        print("Training time,Training min,Training max,Training mean,Training std,Testing time,Testing min,Testing max,Testing mean,Testing std", file=log_file)

        log_file.close()