import numpy as np
from components.episode_buffer import EpisodeBatch
import torch as th


class ESLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.n_agents = args.n_agents
        self.scheme = scheme
        self.logger = logger
        self.input_shape = self._get_input_shape(scheme)
        self.params = list(mac.parameters())
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Prepare structures
        args = self.args
        n_agents = self.n_agents
        agent = self.mac.agent

        # For each agent prepare noises for weights perturbation
        sigma_epsilons = []
        for i in range(0, n_agents):
            sigma_epsilons.append(np.random.normal(args.norm_mean, args.norm_stdev, self.input_shape))
        sigma_epsilons *= args.sigma

        # Add noise to agents' current parameters
        for i in range(0, n_agents):
            if self.args.agent == "rnn":
                index = self.input_shape
                agent.fc1.weight += sigma_epsilons[i][:index]
                agent.rnn.weigth += sigma_epsilons[i][index:index + args.rnn_hidden_dim]
                index += args.rnn_hidden_dim
                agent.fc2.weight += sigma_epsilons[i][index:]

        # Compute outputs for each agent
        agent_outs = self.mac.forward(batch, t=t)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
