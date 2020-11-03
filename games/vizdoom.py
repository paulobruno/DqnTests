from games.game_wrapper import Game
import vizdoom
import itertools as it


class VizDoom(Game):

    def __init__(self, game):
        self.game = game

    @staticmethod
    def configure_from_args(args):
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

    def get_state(self):
        return self.game.get_state().screen_buffer

    def action_space(self):
        return [list(a) for a in it.product([0, 1], repeat=self.game.get_available_buttons_size())]

    def make_action(self, action, repeat=None):
        if repeat is not None:
            return self.game.make_action(action, repeat)
        else:
            return self.game.make_action(action)

    def is_episode_finished(self):
        return self.game.is_episode_finished()

    def new_episode(self):
        return self.game.new_episode()

    def get_total_reward(self):
        return self.game.get_total_reward()

    def set_seed(self, seed):
        self.game.set_seed(seed)

    def close(self):
        self.game.close()

    def init(self):
        self.game.init()

    def make_visible(self):
        self.game.set_window_visible(True)
        self.game.set_mode(vizdoom.Mode.ASYNC_PLAYER)

    def play(self, action, repeat=0):
        self.game.set_action(action)
        for _ in range(repeat):
            self.game.advance_action()
