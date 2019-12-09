from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class EsMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.agents = []
        self.new_agent = None
        self.agent_set = [False for _ in range(args.batch_size_run)]

        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def gradient_ascent_step(self, summed):
        # Perform gradient ascent step
        with th.no_grad():
            j = 0
            for param in self.new_agent.parameters():
                param.data += summed[j].to(param.device)
                j += 1

        # Set new agents net to updated one
        for agent in self.agents:
            agent.load_state_dict(self.new_agent.state_dict())

    def forward(self, ep_batch, t, test_mode=False):
        # TODO: self.agents must be of type es_rnn. Insert implementation error if not provided
        epsilons = ep_batch["epsilons"]
        agent_inputs = self._build_inputs(ep_batch, t)
        device = "cuda" if self.args.use_cuda else "cpu"
        agent_outs = th.tensor([], dtype=agent_inputs[0].dtype, device=device)

        for i in range(self.args.batch_size_run):
            # Do it for every species of the population
            eps = epsilons[i]
            if not self.agent_set[i]:
                self.alter_network(i, eps)
                self.agent_set[i] = True
            # Get current input
            start = self.n_agents * i
            finish = (self.n_agents * i) + self.n_agents
            current_input = agent_inputs[start:finish]
            agent_i_out, self.hidden_states[i] = self.agents[i](current_input, self.hidden_states[i], ep_batch, i)
            agent_outs = th.cat((agent_outs, agent_i_out))
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def alter_network(self, i, eps):
        # Perturb weights
        with th.no_grad():
            j = 0
            for param in self.agents[i].parameters():
                param.data += eps[j].to(param.device)
                j += 1

    def weight_decay(self, amt, t_env):
        perc = 1 - amt
        with th.no_grad():
            for param in self.new_agent.parameters():
                temp = param.data
                temp *= perc
                if False not in th.isnan(temp):
                    self.logger.console_logger.warning("Skipping NaN update for network decay at iteration {}.".format(t_env))
                else:
                    param *= perc

    def reset(self):
        for agent in self.agents:
            agent.load_state_dict(self.new_agent.state_dict())
        self.agent_set = [False] * len(self.agent_set)

    def init_hidden(self, batch_size):
        self.hidden_states = [agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1) for agent in self.agents]  # bav

    def cuda(self):
        for agent in self.agents:
            agent.cuda()
        self.new_agent.cuda()

    def save_models(self, path):
        th.save(self.new_agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        for agent in self.agents:
            agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.new_agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _build_agents(self, input_shape):
        self.new_agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        for i in range(self.args.batch_size_run):
            self.agents.append(agent_REGISTRY[self.args.agent](input_shape, self.args))
            self.agents[i].load_state_dict(self.new_agent.state_dict())

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
