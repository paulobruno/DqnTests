from __future__ import absolute_import, print_function, division, unicode_literals


import argparse

parser = argparse.ArgumentParser(
    description="Run a frozen agent in a given map. It is useful to quickly visualize a map.")

parser.add_argument(
    "config_file",
    help="CFG file with settings of the ViZDoom scenario.")
parser.add_argument(
    "-n", "--num-games",
    type=int,
    metavar="N",
    default=1,
    help="Number of games to play. [default=1]")
parser.add_argument(
    "-r", "--random-agent",
    action="store_true",
    help="Run a random agent instead of a frozen one.")
parser.add_argument(
    "-s", "--spectator",
    action="store_true",
    help="Run the map in spectator mode.")
    
args = parser.parse_args()


import vizdoom
import random
import itertools as it
from time import sleep


FRAME_SKIP = 4


def initialize_vizdoom(args):
    game = vizdoom.DoomGame()
    game.load_config(args.config_file)
    game.set_window_visible(True)
    game.set_mode(vizdoom.Mode.SPECTATOR if args.spectator else vizdoom.Mode.PLAYER)
    game.set_screen_format(vizdoom.ScreenFormat.GRAY8)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_640X480)
    game.init()
    return game


if __name__ == "__main__":

    game = initialize_vizdoom(args)
    
    num_actions = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=num_actions)]

    for i in range(args.num_games):
        game.new_episode()

        while not game.is_episode_finished():                
            game.set_action(random.choice(actions) if args.random_agent else actions[0])

            for _ in range(FRAME_SKIP):
                game.advance_action()

        sleep(1.0)
        score = game.get_total_reward()
        print("Score ep. " + str(i) + ": " + "{:2.0f}".format(score))
        
    game.close()
