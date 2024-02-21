"""
Source: https://github.com/openai/baselines/blob/master/baselines/ddpg/ddpg.py
"""
import numpy as np


class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

    def clear(self):
        self.start = 0
        self.length = 0
        self.data[:] = 0  # unnecessary, not freeing any memory, could be slow


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class Memory(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.terminals = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        # batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.random_integers(low=0, high=self.nb_entries-1, size=batch_size)

        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)

        if next_actions is not None:
            return states_batch, actions_batch, rewards_batch, next_states_batch, next_actions, terminals_batch
        else:
            return states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, training=True):
        if not training:
            return

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.next_actions.clear()
        self.terminals.clear()

    @property
    def nb_entries(self):
        return len(self.states)


class MemoryV2(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False, time_steps=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.time_steps = RingBuffer(limit, shape=(1,)) if time_steps else None
        self.terminals = RingBuffer(limit, shape=(1,))

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        #batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.choice(self.nb_entries, size=batch_size)
        # batch_idxs = random_machine.choice(self.nb_entries, weights=[i/self.nb_entries for i in range(self.nb_entries)], size=batch_size)

        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)
        time_steps = self.time_steps.get_batch(batch_idxs) if self.time_steps is not None else None

        ret = [states_batch, actions_batch, rewards_batch, next_states_batch]
        if next_actions is not None:
            ret.append(next_actions)
        ret.append(terminals_batch)
        if time_steps is not None:
            ret.append(time_steps)
        return tuple(ret)

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, time_steps=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions is not None:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)
        if self.time_steps is not None:
            self.time_steps.append(time_steps)

    @property
    def nb_entries(self):
        return len(self.states)


class MemoryNStepReturns(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False, time_steps=False, n_step_returns=False):
        self.limit = limit

        self.states = RingBuffer(limit, shape=observation_shape)
        self.actions = RingBuffer(limit, shape=action_shape)
        self.rewards = RingBuffer(limit, shape=(1,))
        self.next_states = RingBuffer(limit, shape=observation_shape)
        self.next_actions = RingBuffer(limit, shape=action_shape) if next_actions else None
        self.time_steps = RingBuffer(limit, shape=(1,)) if time_steps else None
        self.terminals = RingBuffer(limit, shape=(1,))
        self.n_step_returns = RingBuffer(limit, shape=(1,)) if n_step_returns else None

    def sample(self, batch_size, random_machine=np.random):
        # Draw such that we always have a proceeding element.
        #batch_idxs = random_machine.random_integers(self.nb_entries - 2, size=batch_size)
        batch_idxs = random_machine.choice(self.nb_entries, size=batch_size)
        # batch_idxs = random_machine.choice(self.nb_entries, weights=[i/self.nb_entries for i in range(self.nb_entries)], size=batch_size)

        '''states_batch = array_min2d(self.states.get_batch(batch_idxs))
        actions_batch = array_min2d(self.actions.get_batch(batch_idxs))
        rewards_batch = array_min2d(self.rewards.get_batch(batch_idxs))
        next_states_batch = array_min2d(self.next_states.get_batch(batch_idxs))
        terminals_batch = array_min2d(self.terminals.get_batch(batch_idxs))'''
        states_batch = self.states.get_batch(batch_idxs)
        actions_batch = self.actions.get_batch(batch_idxs)
        rewards_batch = self.rewards.get_batch(batch_idxs)
        next_states_batch = self.next_states.get_batch(batch_idxs)
        next_actions = self.next_actions.get_batch(batch_idxs) if self.next_actions is not None else None
        terminals_batch = self.terminals.get_batch(batch_idxs)
        time_steps = self.time_steps.get_batch(batch_idxs) if self.time_steps is not None else None
        n_step_returns = self.n_step_returns.get_batch(batch_idxs) if self.n_step_returns is not None else None

        ret = [states_batch, actions_batch, rewards_batch, next_states_batch]
        if next_actions is not None:
            ret.append(next_actions)
        ret.append(terminals_batch)
        if time_steps is not None:
            ret.append(time_steps)
        if n_step_returns is not None:
            ret.append(n_step_returns)
        return tuple(ret)

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, time_steps=None,
               n_step_return=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        if self.next_actions is not None:
            self.next_actions.append(next_action)
        self.terminals.append(terminal)
        if self.time_steps is not None:
            assert time_steps is not None
            self.time_steps.append(time_steps)
        if self.n_step_returns is not None:
            assert n_step_return is not None
            self.n_step_returns.append(n_step_return)

    @property
    def nb_entries(self):
        return len(self.states)


class SumTree:
    '''https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay.'''
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.data = np.zeros(capacity, dtype=object)  # storing experience
        self.tree = np.zeros(
            2 * capacity - 1)  # storing pripority + parental nodes. If you have n elements in the bottom, then you need n + (n-1) nodes to construct a tree.
        self.n_entries = 0
        self.overwrite_start_flag = False  # record whether N_entry > capacity

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1  # starting write from the first element of the bottom layer,
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            self.overwrite_start_flag = True
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        # --------original----------
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

    def max_priority(self):
        return np.max(self.tree[self.capacity - 1:])

    # ---- modified : later to avoid min_prob being overwritten ----
    def min_prob(self):
        #        p_min = float(np.min(self.tree[self.capacity-1:self.n_entries+self.capacity-1])) / self.total()
        if self.overwrite_start_flag == False:
            p_min = float(np.min(self.tree[self.capacity - 1:self.n_entries + self.capacity - 1])) / self.total()
            self.p_min_history = p_min
        elif self.overwrite_start_flag == True:
            p_min = min(float(np.min(self.tree[self.capacity - 1:self.n_entries + self.capacity - 1])) / self.total(),
                        self.p_min_history)
            self.p_min_history = min(p_min, self.p_min_history)
        return p_min



class Prioritized_Experience_Replay(object):
    def __init__(self, limit, observation_shape, action_shape, next_actions=False, PER_NO_IS=False, alpha=0.6, beta=0.4, dtype='float32'):
        self.sum_tree = SumTree(limit)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta)/100000.
        self.current_length = 0
        self.state_dim = observation_shape
        self.action_dim = action_shape
        self.dtype = dtype
        self.PER_NO_IS = PER_NO_IS

    @property
    def nb_entries(self):
        return len(self.states)

    def append(self, state, action, reward, next_state, next_action=None, terminal=False, training=True):
        if not training:
            return

        if self.current_length == 0:
            priority = 1.0
        else:
            priority = self.sum_tree.max_priority()
        self.current_length = self.current_length + 1
        # priority = priority ** self.alpha
        # -------------modified for efficient storage--------------
        #        experience = (state, np.array([action]), np.array([reward]), next_state, done)
        experience = (state, action, reward, next_state, terminal)
        self.sum_tree.add(priority, experience)

    def sample(self, batch_size, random_machine=np.random):
        # ----------------To modify: sampling for efficient memory storage------------------
        batch_idx, batch, priorities = [], [], []
        segment = self.sum_tree.total() / batch_size
        p_sum = self.sum_tree.tree[0]

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            # s = random.uniform(a, b)
            s = random_machine.uniform(a, b)
            idx, p, data = self.sum_tree.get(s)
            #            print(s, idx-1048576, p)
            batch_idx.append(idx)
            batch.append(data)
            priorities.append(p)

        if not self.PER_NO_IS:
            prob = np.array(priorities) / p_sum
            IS_weights = np.power(self.sum_tree.n_entries * prob, -self.beta)
            max_weight = np.power(self.sum_tree.n_entries * self.sum_tree.min_prob(), -self.beta)
            IS_weights /= max_weight
        else:
            IS_weights = np.ones_like(priorities)

        state_batch = np.empty((batch_size, *self.state_dim), dtype=self.dtype)
        action_batch = np.empty((batch_size, *self.action_dim), dtype=self.dtype)
        reward_batch = np.empty((batch_size, 1), dtype=self.dtype)
        next_state_batch = np.empty((batch_size, *self.state_dim), dtype=self.dtype)
        done_batch = np.empty((batch_size, 1), dtype=self.dtype)
        for i, transition in enumerate(batch):
            state, action, reward, next_state, done = transition
            state_batch[i] = state
            action_batch[i] = action
            reward_batch[i] = reward
            next_state_batch[i] = next_state
            done_batch[i] = done

        self.beta = min(1., self.beta + self.beta_increment)
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch), batch_idx, IS_weights


    def update_priority(self, idx, td_error):
        priority = np.power(td_error, self.alpha)
        self.sum_tree.update(idx, priority)

    def __len__(self):
        return self.current_length
