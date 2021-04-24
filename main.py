from __future__ import absolute_import, print_function, division, unicode_literals
from utils.args import log_params, build_args

args = build_args()
    
import os # to check if a model folder exists

import tensorflow as tf
from tensorflow import keras

# from agent.dueling_agent import Agent
from agent.reinforce_agent import ReinforceAgent as Agent
from games.vizdoom import VizDoom

# limit gpu usage
# by default, tensorflow allocates all available memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


params = {
    "learning_rate": 0.0001,
    "discount_factor": 0.99,
    "replay_memory_size": 10000,
    "batch_size": 64,
    "dropout_prob": 0.3,
    "train_episodes_per_epoch": 20,
    "test_episodes_per_epoch": 200,
    "resolution": (64, 128),
    "frame_repeat": 5,
    "episodes_to_watch": 10,
    "num_epochs": 20,
    "should_save_model": args.save_model,
    "game": VizDoom(VizDoom.configure_from_args(args))
}


if __name__ == '__main__':

    # create vizdoom game
    agent = Agent(**params)

    # args.load_model = True

    # load or create new model
    if (args.load_model and os.path.isdir(args.model_folder)):
        print("Loading model from " + args.model_folder + ".")
        agent.net.load("{}/models/model.h5".format(args.model_folder))
        agent.net.load_checkpoits("{}/checkpoints/".format(args.model_folder))
    else:
        if not os.path.isdir(args.model_folder):
            print("No folder was found in " + args.model_folder + ".")

    if args.show_model:
        agent.net.net.model.summary()
        keras.utils.plot_model(agent.net.net.model, args.show_model + ".png", show_shapes=True)
        
    log_params(args, params)

    agent.run(args.save_model_interval, args.model_folder)

    agent.eval()

    


