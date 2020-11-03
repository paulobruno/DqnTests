
from games.game_wrapper import Game
import gym


class GymGame(Game):

    def __init__(self, game):
        self.game = game
        self.acc_reward = 0
        self.should_clear = False
        self.actual_state = None
        self.is_done = None
        self.info = None
        self.is_visible = True
        self.count = 0

    @staticmethod
    def configure_from_args(args):
        print("[1.] Initializing GYM: {}...".format('CartPole-v0'))
        game = gym.make('CartPole-v0')
        game.reset()
        print("[.1] ... GYM initialized.")
        return game

    def get_state(self):
        if self.actual_state is None:
            self.actual_state = self.game.reset()
        return self.actual_state

    def make_action(self, action, repeat=None):

        self.count += 1

        if self.should_clear:
            self.acc_reward = 0

        if self.is_visible:
            self.game.render()

        observation, reward, done, info = self.game.step(action)

        self.actual_state = observation
        self.is_done = done
        self.info = info

        self.acc_reward += reward
        self.should_clear = done

        print(self.count)

        return reward

    def is_episode_finished(self):
        return self.is_done

    def new_episode(self):
        return self.game.reset()

    def get_total_reward(self):
        return self.acc_reward

    def set_seed(self, s):
        self.game.seed(s)

    def action_space(self):
        return range(self.game.env.action_space.n)

    def close(self):
        self.game.close()

    def init(self):
        self.game.reset()

    def make_visible(self):
        self.is_visible = True