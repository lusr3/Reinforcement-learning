from typing import (
    Tuple,
)

import torch
import random
from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
    Priority,
    Batchidx,
    Batchweigths
)

class SumTree(object):

    __index = 0
    __size = 0

    def __init__(self, tree, capacity, states, actions, rewards, dones,
    ) -> None:
        # priority storage in tree
        self.__tree = tree
        self.__capacity = capacity

        # data
        self.__m_states = states
        self.__m_actions = actions
        self.__m_rewards = rewards
        self.__m_dones = dones

    # add new node
    def add(self, priority, state, action, reward, done
    ) -> None:
        tree_index = self.__index + self.__capacity - 1
        self.__m_states[self.__index] = state
        self.__m_actions[self.__index, 0] = action
        self.__m_rewards[self.__index, 0] = reward
        self.__m_dones[self.__index, 0] = done

        # update father node value
        self.update(tree_index, priority)

        self.__index += 1
        self.__size = max(self.__size, self.__index)
        self.__index %= self.__capacity

    # update priority node by index and priority
    def update(self, tree_index, priority
    ) -> None:
        change = priority - self.__tree[tree_index]
        self.__tree[tree_index] = priority
        # 向父节点传递更新
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.__tree[tree_index] += change

    # return one sample
    def get_leaf(
        self,
        value : int
    ) -> Tuple[
            int,
            Priority,
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        cur_idx = 0
        leaf_idx = 0
        while True:
            l_child = 2 * cur_idx + 1
            r_child = l_child + 1
            # find the leaf node
            if l_child >= len(self.__tree):
                leaf_idx = cur_idx
                break
            # less to left, more to right
            # minus node value because of accumalation
            elif value < self.__tree[l_child]:
                cur_idx = l_child
            else:
                value -= self.__tree[l_child]
                cur_idx = r_child
        data_idx = leaf_idx - self.__capacity + 1
        return leaf_idx, self.__tree[leaf_idx], self.__m_states[data_idx, :4], \
        self.__m_actions[data_idx], self.__m_rewards[data_idx], self.__m_states[data_idx, 1:], \
        self.__m_dones[data_idx]
    
    # get sum of priority
    def priority_sum(self):
        return self.__tree[0]

    def size(self) -> int:
        return self.__size


class ReplayMemory(object):

    eps = 0.01          # make it possible for each sample to be selected
    alpha = 0.6         # adjust the degree of priority
    max_err = 1.

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
            full_sink: bool = True,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__channels = channels

        sink = lambda x: x.to(device) if full_sink else x
        self.__m_states = sink(torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8))
        self.__m_actions = sink(torch.zeros((capacity, 1), dtype=torch.long))
        self.__m_rewards = sink(torch.zeros((capacity, 1), dtype=torch.int8))
        self.__m_dones = sink(torch.zeros((capacity, 1), dtype=torch.bool))

        # sumtree
        self.tree = sink(torch.zeros(2 * capacity - 1))
        self.sumtree = SumTree(self.tree, self.__capacity, self.__m_states, self.__m_actions, self.__m_rewards, self.__m_dones)


    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        priority = self.max_err
        self.sumtree.add(priority, folded_state, action, reward, done)

    def sample(self, batch_size: int) -> Tuple[
            Batchidx,
            Batchweigths,
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        psum = self.sumtree.priority_sum()
        b_idx = torch.zeros(batch_size, 1, dtype=torch.long)
        b_state = torch.zeros(batch_size, self.__channels - 1, 84, 84, dtype=torch.uint8)
        b_next = torch.zeros(batch_size, self.__channels - 1, 84, 84, dtype=torch.uint8)
        b_action = torch.zeros(batch_size, 1, dtype=torch.long)
        b_reward = torch.zeros(batch_size, 1, dtype=torch.int8)
        b_done = torch.zeros(batch_size, 1, dtype=torch.bool)

        # segmentation
        dseg = psum / batch_size
        # find leaf node at every segment
        for idx in range(batch_size):
            # intervel
            left, right = dseg * idx, dseg * (idx + 1)
            value = random.uniform(left, right)
            t_idx, pri, c_state, c_action, c_reward, c_next, c_done = \
                self.sumtree.get_leaf(value=value)
            b_idx[idx, 0] = t_idx
            b_state[idx] = c_state
            b_next[idx] = c_next
            b_action[idx] = c_action
            b_reward[idx] = c_reward
            b_done[idx] = c_done
        
        b_idx = b_idx.to(self.__device)
        b_state = b_state.to(self.__device).float()        # (batch_size, 4, 84, 84)
        b_next = b_next.to(self.__device).float()         # (batch_size, 4, 84, 84)
        b_action = b_action.to(self.__device)                  # (batch_size, 1)
        b_reward = b_reward.to(self.__device).float()          # (batch_size, 1)
        b_done = b_done.to(self.__device).float()              # (batch_size, 1)

        return b_idx, b_state, b_action, b_reward, b_next, b_done
    
    # update sumtree after train
    def batch_update(self, b_idx, b_tderr) -> None:
        b_tderr += self.eps
        b_tderr = torch.pow(b_tderr, self.alpha)
        self.max_err = max(self.max_err, b_tderr.max())
        for idx, err in zip(b_idx, b_tderr):
            self.sumtree.update(idx, err)

    def __len__(self) -> int:
        return self.sumtree.size()

