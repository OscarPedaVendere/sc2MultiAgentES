import torch.nn as nn
import torch.nn.functional as F
import torch as th


class EsRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EsRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, ep_batch):
        pop_size = ep_batch.batch_size
        extrap = [i[0] for i in ep_batch["epsilons"]]
        epsilons = [i.item() for i in extrap[1]]
        n_agents = int(inputs.shape[0] / pop_size)
        q = th.tensor([], dtype=inputs.dtype)
        h = th.tensor([], dtype=inputs.dtype)
        # Save current network's parameters
        initial_params = self.state_dict()

        # Do it for every species of the population
        for i in range(0, pop_size):
            # Get current input
            start = n_agents * i
            finish = (n_agents * i) + n_agents
            current_input = inputs[start:finish]

            # Perturb weights
            x_i = epsilons[i] * self.args.sigma
            with th.no_grad():
                for param in self.parameters():
                    param.__add__(x_i)

            # Forward pass
            x = F.relu(self.fc1(current_input))
            h_tot = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h_in = h_tot[:n_agents]
            h_fin = self.rnn(x, h_in)
            h = th.cat((h, h_fin))
            q = th.cat((q, self.fc2(h_fin)))

            # Reset previous network parameters
            self.load_state_dict(initial_params)

        return q, h
