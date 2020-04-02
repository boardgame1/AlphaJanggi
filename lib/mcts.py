import random, math
import numpy as np

from lib import game, model, actionTable

import torch.nn.functional as F


class MCTS:
    def __init__(self, c_puct=1.0):
        self.c_puct = c_puct
        self.visit_count = {}
        self.value = {}
        self.value_avg = {}
        self.probs = {}

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()

    def __len__(self):
        return len(self.value)

    def find_leaf(self, state_int, player, step):
        states = []
        actions = []
        cur_state = state_int
        cur_player = player
        value = None

        while value is None and not self.is_leaf(cur_state):
            states.append(cur_state)

            counts = self.visit_count[cur_state]
            total_sqrt = math.sqrt(sum(counts))
            probs = self.probs[cur_state]
            values_avg = self.value_avg[cur_state]

            movel, okingp = game.possible_moves(cur_state, cur_player, step)
            alen = len(actions)
            if alen<1:
                noises = np.random.dirichlet([0.17] * len(movel))
            max_score = -np.inf
            chList = actionTable.choList if cur_player < 1 else actionTable.hanList
            for i, m in enumerate(movel):
                idx = chList.index(m)
                if m % 100 == okingp: aidx = idx; break
                score = values_avg[idx] + self.c_puct * (probs[idx] if alen else
                                                         0.75 * probs[idx] + 0.25 * noises[i]) * total_sqrt / (
                                    1 + counts[idx])
                if score > max_score: max_score = score; aidx = idx
            action = chList[aidx]
            actions.append(aidx)
            cur_state, won = game.move(cur_state, action, step)
            if won>0:
                # if somebody won the game, the value of the final state is -1 (as it is on opponent's turn)
                value = -1.0 if won - 1 == cur_player else 1.0
            cur_player = 1-cur_player
            step += 1

        if value != None: value *= 1-(game.MAX_TURN-step)/1000

        return value, cur_state, step, states, actions

    def is_leaf(self, state_int):
        return state_int not in self.probs

    def search_batch(self, count, batch_size, state_int, player, net, step, device="cpu"):
        for _ in range(count):
            self.search_minibatch(batch_size, state_int, player, net, step, device)

    def search_minibatch(self, count, state_int, player, net, step, device="cpu"):
        backup_queue = []
        expand_states = []
        expand_steps = []
        expand_queue = []
        planned = set()
        for _ in range(count):
            value, leaf_state, leaf_step, states, actions = self.find_leaf(state_int, player, step)
            if value is not None:
                backup_queue.append((value, states, actions))
            else:
                if leaf_state not in planned:
                    planned.add(leaf_state)
                    leaf_state_lists = game.decode_binary(leaf_state)
                    expand_states.append(leaf_state_lists)
                    expand_steps.append(leaf_step)
                    expand_queue.append((leaf_state, states, actions))

        if expand_queue:
            batch_v = model.state_lists_to_batch(expand_states, expand_steps, device)
            logits_v, values_v = net(batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            values = values_v.data.cpu().numpy()[:, 0]
            probs = probs_v.data.cpu().numpy()

            for (leaf_state, states, actions), value, prob in zip(expand_queue, values, probs):
                self.visit_count[leaf_state] = [0] * actionTable.AllMoveLength
                self.value[leaf_state] = [0.0] * actionTable.AllMoveLength
                self.value_avg[leaf_state] = [0.0] * actionTable.AllMoveLength
                self.probs[leaf_state] = prob
                backup_queue.append((value, states, actions))

        for value, states, actions in backup_queue:
            cur_value = -value
            for state_int, action in zip(states[::-1], actions[::-1]):
                self.visit_count[state_int][action] += 1
                self.value[state_int][action] += cur_value
                self.value_avg[state_int][action] =\
                    self.value[state_int][action] / self.visit_count[state_int][action]
                cur_value = -cur_value

    def get_policy_value(self, state_int, movel, cur_player, tau=1):
        counts = [0] * actionTable.AllMoveLength
        chList = actionTable.choList if cur_player < 1 else actionTable.hanList
        for m in movel:
            idx = chList.index(m)
            counts[idx] = self.visit_count[state_int][idx]
        total = sum(counts)
        if tau == 0 or total < 1:
            probs = [0.0] * actionTable.AllMoveLength
            probs[np.argmax(counts) if total > 0 else chList.index(movel[random.randrange(0, len(movel))])] = 1.0
        else:
            counts = [count ** (1.0 / tau) for count in counts]
            probs = [count / total for count in counts]
        values = self.value_avg[state_int]
        return probs, values
