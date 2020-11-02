from __future__ import absolute_import, print_function, division, unicode_literals


# parse command line arguments
import argparse

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

args = parser.parse_args()
    

import vizdoom

import os # to check if a model folder exists

import tensorflow as tf
from tensorflow import keras

from agent import Agent

# limit gpu usage
# by default, tensorflow allocates all available memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Q-learning settings
learning_rate = 0.0001
discount_factor = 0.99
replay_memory_size = 10000

# NN learning settings
batch_size = 64
dropout_prob = 0.3

# Training regime
episodes_per_epoch = 200
test_episodes_per_epoch = 200

# Other parameters
resolution = (48, 64)
frame_repeat = 8
episodes_to_watch = 5



def initialize_vizdoom(args):
    print("[1.] Initializing ViZDoom...")
    game = vizdoom.DoomGame()
    game.load_config(args.config_file)
    game.set_window_visible(args.enable_training_view)
    game.set_mode(vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.init()
    print("[.1] ... ViZDoom initialized.")
    return game


if __name__ == '__main__':

    # create vizdoom game
    game = initialize_vizdoom(args)
    agent = Agent(num_epochs=args.num_epochs, batch_size=batch_size, game=game, resolution=resolution,
                  should_save_model=True, replay_memory_size=replay_memory_size,
                  train_episodes_per_epoch=episodes_per_epoch, test_episodes_per_epoch=test_episodes_per_epoch)

    args.load_model = False
    
    # load or create new model
    if (args.load_model and os.path.isdir(args.model_folder)):
        print("Loading model from " + args.model_folder + ".")
        agent.dqn.load(args.model_folder)
    else:
        if not os.path.isdir(args.model_folder):
            print("No folder was found in " + args.model_folder + ".")

        
    if args.show_model:
        agent.dqn.dqn.model.summary()
        keras.utils.plot_model(agent.dqn.dqn.model, args.show_model + ".png", show_shapes=True)
        

    # open file to save resultsMap: line
    if args.log_file:
        log_file = open(args.log_file, "w", buffering=1)
        print("Map,{}".format(args.config_file), file=log_file)
        print("Resolution,{}".format(resolution), file=log_file)
        print("Frame Repeat,{}".format(frame_repeat), file=log_file)
        print("Learning Rate,{}".format(learning_rate), file=log_file)
        print("Discount,{}".format(discount_factor), file=log_file)
        print("Replay Memory,{}".format(replay_memory_size), file=log_file)
        print("Batch Size,{}".format(batch_size), file=log_file)
        print("Dropout,{}".format(dropout_prob), file=log_file)
        print("Epochs,{}".format(args.num_epochs), file=log_file)
        print("Training Episodes,{}".format(episodes_per_epoch), file=log_file)
        print("Testing Episodes,{}".format(test_episodes_per_epoch), file=log_file)
        print("Training time,Training min,Training max,Training mean,Training std,Testing time,Testing min,Testing max,Testing mean,Testing std", file=log_file)
        

    agent.run(args.save_model_interval, args.model_folder)


    # close file with results
    if args.log_file:
        log_file.close()

    agent.eval()

    


