from components.episode_buffer import EpisodeBatch
import torch as th


class ESLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.n_agents = args.n_agents
        self.scheme = scheme
        self.logger = logger
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        # terminated = batch["terminated"][:, :-1].float()
        # mask = batch["filled"][:, :-1].float()
        # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        epsilons = batch["epsilons"]

        # Prepare structures
        n = self.args.batch_size_run
        rews = th.sum(rewards, dim=1)
        adv = (rews - rews.mean()) / rews.std()
        skip_update = False not in (adv == 0)

        if not skip_update:
            # Compute fraction * sum
            fraction = self.args.alpha / (n * self.args.sigma)
            for i in range(len(epsilons)):
                curr_eps = epsilons[i]
                f_i = adv[i].item() if not th.isnan(adv[i]).item() else 0
                for j in curr_eps:
                    j *= f_i

            # Compute sum of each epsilon
            summed = epsilons[0]
            for i in range(len(epsilons)):
                k = 0
                for j in epsilons[i]:
                    summed[k] += j
                    k += 1

            # TODO: Fix this implementation/calculation error
            # Multiply times fraction
            for e in summed:
                e *= fraction

            # TODO: self.mac must be of type es_mac. Insert implementation error if not provided
            for e in epsilons:
                for j in e:
                    if False not in th.isnan(j):
                        self.logger.console_logger.warning("Skipping NaN update for network at episode {}".format(episode_num))
                        return

            # Perform a gradient ascent step
            self.mac.gradient_ascent_step(summed)

            # Reset the controller for the next episode
            self.mac.reset()

        if self.args.weight_decay:
            should_decay = False
            if self.args.decay_limit == 0:  # No checks for upper limit
                if t_env >= self.args.decay_start:
                    should_decay = True
            else:   # Check for upper limit
                if t_env in range(self.args.decay_start, self.args.decay_limit):
                    should_decay = True
            if should_decay:
                self.mac.weight_decay(self.args.decay_amount, t_env, episode_num)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def cuda(self):
        self.mac.cuda()

    def save_models(self, path):
        self.mac.save_models(path)

    def load_models(self, path):
        self.mac.load_models(path)
