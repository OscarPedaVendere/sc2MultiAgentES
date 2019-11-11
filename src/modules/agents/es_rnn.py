from .rnn_agent import RNNAgent
import torch.nn.functional as F


class ESRNNAgent(RNNAgent):
    def forward(self, inputs, hidden_state, ep_batch, i):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state
        if len(hidden_state.size()) > 2:
            h_in = hidden_state[i]
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
