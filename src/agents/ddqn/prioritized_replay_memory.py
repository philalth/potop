"""
Module for prioritized replay memory
"""
import random
from collections import namedtuple, deque
from typing import List

import numpy as np

PrioritizedTransition = namedtuple('PrioritizedTransition', ('priority', 'data'))
WeightedTransition = namedtuple('WeightedTransition', ('weight', 'original_index', 'data'))


class ReplayMemory:
    """Experience Buffer for Deep RL Algorithms."""

    def __init__(self, size):
        self.transitions = deque(maxlen=size)
        self.episodes = []
        self.counter = 0
        self.counter_at_last_episode = 0

    def save(self, transition):
        """Save a transition."""
        self.transitions.append(transition)
        self.counter += 1

    def sample_batch(self, minibatch_size):
        """Sample from batch with a given minibatch_size."""
        if self.size() > minibatch_size:
            return random.sample(self.transitions, minibatch_size)
        return list(self.transitions)

    def sample_day(self) -> List[List[np.array]]:
        """Sample the observations from a whole day for the Q-mix.

        Returns:
            days: a list of all days in the memory.
                  a single day in this list is a list of transition that occurred on this day
        """
        days = []
        for transition in self.transitions:
            assert len(transition) == 5
            if transition[0][0][5].item() == 0.0:  # Datetime of old_state of observation
                days.append([])
            if len(days) == 0:
                continue
            days[-1].append(transition)

        return days

    def sample_episode(self) -> List[List[np.array]]:
        """Sample the observations from a whole episode for the Q-mix.

        The memory size needs to be as least as big as the number of transitions per episode!
        Oherwise earlier transitions will be missing as they cannot be archived by the
        signal_starting_episode method after each episode.

        Returns:
            episodes: a list of all episodes in the memory.
                      a single episode in this list is a list of all transitions in that episode
        """
        return self.episodes

    def signal_starting_episode(self):
        """The main loop can tell the memory breaks between episodes here."""
        if self.counter != 0:
            if self.counter - self.counter_at_last_episode > len(self.transitions):
                # If more steps happened since the last episode than the size of the memory
                self.episodes.append(list(self.transitions))
            else:
                num_obs = self.counter - self.counter_at_last_episode
                self.episodes.append(list(self.transitions)[-num_obs:])
            self.counter_at_last_episode = self.counter

    def clear(self):
        """Clear transitions."""
        self.transitions.clear()

    def size(self):
        """Return the number of transitions."""
        return len(self.transitions)


class PrioritizedReplayMemory:
    """Implements a prioritized replay buffer as described in Schaul et al. (2016).

    This implementation uses rank-based probabilities and
    a linked list data structure that is sorted each time a batch is sampled.

    Sampling from equal probability bins guarantees stochastic prioritization.
    Batches also contain importance sampling (IS) weights to correct for bias due to
    unequal sampling.

    References:
        Schaul, Quan, Antonoglou and Silver (2016): Prioritizied Experience Replay;
            https://arxiv.org/abs/1511.05952
    """

    def __init__(self, size: int, batch_size: int, alpha: float = 0.7, beta: float = 1.0):
        """Create a new replay memory.

        Parameters:
            size: maximum capacity
            batch_size: only batches of this size can later be drawn from the memory
            alpha: hyperparameter between 0.0 and 1.0
                Determines how much observations with high td_error should be preferred.
            beta: hyperparameter between 0.0 and 1.0
                Determines how small weights for transitions with high probability should be
        """
        self.transitions = deque(
            maxlen=size)  # Automatically deletes old observations if maxlen is reached
        self.batch_size = batch_size  # Batch size needs to be fixed
        # for the underlying data structure
        self.bins = self.precompute_rank_bins(size, self.batch_size, alpha, beta)
        self.max_td_error = -1_000_000.0

    @staticmethod
    def precompute_rank_bins(size: int, num_bins: int, alpha: float, beta: float):
        """See Schaul et al. (2016), page 4, paragraph 'Implementation'.

        This method will divide pairs of (probability, observation) into bins
        such that the sum of all probabilities in a bin is roughly equal among the bins.
        The batch will then uniformly sample one obseration from each of the bins.
        Observations with higher probabilities will be in smaller bins
        and will therefore be sampled with a higher probability.

        The probabilities are calculated as:
            p_i = 1 / rank(i)
            P_i = p_i ** alpha / Sum_i(p_i ** alpha)
        where rank(i) is the rank of the observation according to the temporal difference error.
        As the bins for these probabilities and a fixed size remain the same,
         the can be precomputed and stored.

        This is not guaranteed to calculate actually {num_bins} filled bins,
        as the first bins are usually a bit too full!
        Fill the sample up randomly to compensate for empty bins at the end.

        Params:
            size: total number of observations that should be divided into bins
                This should be set to len(self.transitions)
            num_bins: number of bins with equal probability that should be created.
                This should be set to self.batch_size
            alpha: hyperparameter between 0 and 1
                0: uniform distribution, 1: most unequal distribution
            beta: hyperparameter between 0 and 1
                0: all weights are 1, 1: smallest weights for transitions
                 with highest probabilities

        Returns:
            bins, a list of lists of pairs (index, is_weight), mapping ranks to bins and weights.
            Sample from bins and apply to sorted list.
        """
        ranks = [1 / (i + 1) ** alpha for i in range(size)]
        total = sum(ranks)
        probs = [rank / total for rank in ranks]
        is_weights = [1 / (size * prob) ** beta for prob in probs]
        # Normalize importance sampling weights
        maximum = max(is_weights)
        is_weights = [weight / maximum for weight in is_weights]

        # Fill bins
        # After this step, the sum of all entries within a bin should be roughly equal
        # But in reality, not all bins get filled as the first bins are too full!
        # Sample randomly more observations to get a total sample of batch_size
        bins = [[] for _ in range(num_bins)]
        prob_per_bin = 1 / num_bins
        cumulative_bin_probs = [0 for _ in range(num_bins)]
        active_bin = 0
        for index, prob in enumerate(probs):  # Ordered iteration over the transitions
            bins[active_bin].append((index, is_weights[index]))
            cumulative_bin_probs[active_bin] += prob
            if cumulative_bin_probs[active_bin] >= prob_per_bin and active_bin < num_bins:
                active_bin += 1

        return bins

    def save(self, transition, td_error=None):
        """Save a transition to the memory as a namedtuple (td_error, transition).

        If td_error is None: must be new experience, gets maximum priority
        so it will appear at least once in the training.
        """
        if td_error is None:
            td_error = self.max_td_error
        elif td_error > self.max_td_error:
            self.max_td_error = td_error
        transition = PrioritizedTransition(priority=td_error, data=transition)
        self.transitions.append(transition)

    # pylint: disable=E0633
    # ODO: investigate why "Attempting to unpack a non-sequence"
    def sample_batch(self):
        """Sample a prioritized batch from the memory.

        The batch will contain one observation from each of the precomputed bins.
        As some of the bins contain fewer observations than others,
        this prioritizes transitions with a higher temporal difference error.

        Returns:
            A list of size self.batch_size from the replay memory.
                Each entry is a namedtuple WeightedTransition(is_weight, original_index,
                transition)
                The original_index (=>indices in self.transitions of the sampled transitions)
                is used for updating the td_error after training!
        """

        sorted_transitions = sorted(enumerate(self.transitions),
                                    reverse=True, key=lambda x: x[1][0])

        # Sample from n=self.batch_size bins with equal probability
        # prob_per_bin = 1 / self.batch_size

        # Sample once from every bin. This guarantees also a stratified batch which should
        # improve training stability
        samples = []
        for ind in range(self.batch_size):
            try:
                index, is_weight = random.sample(self.bins[ind], 1)[0]
                transition = sorted_transitions[index][1].data
                original_index = sorted_transitions[index][0]
                samples.append(WeightedTransition(weight=is_weight,
                                                  original_index=original_index,
                                                  data=transition))

            except (ValueError, IndexError):
                # If bin was empty or replay memory is not full enough yet:
                # sample randomly from whole population
                random_index = random.randint(0, len(self.transitions)-1)
                transition = self.transitions[random_index][1].data
                is_weight = 1 / len(self.transitions)
                samples.append(WeightedTransition(weight=is_weight,
                                                  original_index=random_index,
                                                  data=transition))

        return samples

    def update_td_errors(self, indices, td_errors):
        """Always call this function after training on a batch!

        This will update the priorities of transitions with the newly calculated temporal
        difference errors.
        Priorities will change after the first time of training on it and as the network learns.
        """
        assert len(indices) == len(td_errors)
        for index, _ in enumerate(indices):
            old_transition = self.transitions[indices[index]]
            new_transition = PrioritizedTransition(priority=td_errors[index],
                                                   data=old_transition.data)
            self.transitions[indices[index]] = new_transition
            if td_errors[index] > self.max_td_error:
                self.max_td_error = td_errors[index]

    def clear(self):
        """Clear all transitions."""
        self.transitions.clear()

    def size(self):
        """Return the number of transitions."""
        return len(self.transitions)
