from agent.base_agent import BaseAgent
from memory.relay_memory import ReplayMemory as memory


class AgentRelayMemory(BaseAgent):

    def __init__(self, params):
        super(AgentRelayMemory, self).__init__(params)

        state_shape = (params['replay_memory_size'], params['resolution'][0], params['resolution'][1], self.channel)
        self.memory = memory(state_shape)

    def run_step(self):
        s1 = self.preprocess(self.game.get_state())

        # With probability eps make a random action.
        a = self.get_action(s1)

        reward = self.game.make_action(self.actions[a], self.frame_repeat)

        isterminal = self.game.is_episode_finished()
        s2 = self.preprocess(self.game.get_state()) if not isterminal else None

        # Remember the transition that was just experienced.
        self.memory.store((s1, a, s2, isterminal, reward))

        if self.memory.size > self.batch_size:
            tree_idx, experience = self.memory.sample(self.batch_size)
            self.net.train_step(experience)
