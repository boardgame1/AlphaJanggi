"""
Monte-Carlo Tree Search
"""
import random, math
import numpy as np

from lib import game, model

import torch.nn.functional as F


def hanAction(m):
    a = m // 100;
    a = (9 - a // 9) * 9 + a % 9
    b = m % 100;
    b = (9 - b // 9) * 9 + b % 9
    return a * 100 + b

class MCTS:
    """
    Class keeps statistics for every state encountered during the search
    """
    def __init__(self, c_puct=1.0):
        self.c_puct = c_puct
        # count of visits, state_int -> [N(s, a)]
        self.visit_count = {}
        # total value of the state's action, state_int -> [W(s, a)]
        self.value = {}
        # average value of actions, state_int -> [Q(s, a)]
        self.value_avg = {}
        # prior probability of actions, state_int -> [P(s,a)]
        self.probs = {}

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()

    def __len__(self):
        return len(self.value)

    def find_leaf(self, state_int, player, step):
        """
        Traverse the tree until the end of game or leaf node
        :param state_int: root node state
        :param player: player to move
        :return: tuple of (value, leaf_state, player, states, actions)
        1. value: None if leaf node, otherwise equals to the game outcome for the player at leaf
        2. leaf_state: state_int of the last state
        3. player: player at the leaf node
        4. states: list of states traversed
        5. list of actions taken
        """
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
            # choose action to take, in the root node add the Dirichlet noise to the probs
            alen = len(actions)
            if alen<1:
                noises = np.random.dirichlet([0.17] * len(movel))
            max_score = -np.inf
            for i, m in enumerate(movel):
                if m<1 or m>9999:
                    idx = 90 if m<1 else m-10000+91; ma = m
                    score = values_avg[idx] + self.c_puct * (probs[idx] if alen else
                            0.75 * probs[idx] + 0.25 * noises[i])* total_sqrt / (1 + counts[idx])
                else:
                    ma = hanAction(m) if cur_player > 0 else m
                    if m%100 == okingp: action = m; mam = ma; break
                    idx = ma//100; idx2 = ma%100+95
                    score = values_avg[idx] + values_avg[idx2] + self.c_puct * (probs[idx]+probs[idx2]
                        if alen else 0.75 * probs[idx]+probs[idx2] + 0.25 * noises[i])* total_sqrt /\
                            (1 + counts[idx]+counts[idx2])
                if score>max_score: max_score=score; action = m; mam = ma
            actions.append(mam)
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
        """
        Perform several MCTS searches.
        """
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

        # do expansion of nodes
        if expand_queue:
            batch_v = model.state_lists_to_batch(expand_states, expand_steps, device)
            logits_v, values_v = net(batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            values = values_v.data.cpu().numpy()[:, 0]
            probs = probs_v.data.cpu().numpy()

            # create the nodes
            for (leaf_state, states, actions), value, prob in zip(expand_queue, values, probs):
                self.visit_count[leaf_state] = [0] * model.policy_size
                self.value[leaf_state] = [0.0] * model.policy_size
                self.value_avg[leaf_state] = [0.0] * model.policy_size
                self.probs[leaf_state] = prob
                backup_queue.append((value, states, actions))

        # perform backup of the searches
        for value, states, actions in backup_queue:
            # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
            cur_value = -value
            for state_int, action in zip(states[::-1], actions[::-1]):
                for i in range(2):
                    a1 = 90 if action<1 else action-10000+91 if action>9999 else\
                        action//100 if i<1 else action%100+95
                    if i<1 or a1>94:
                        self.visit_count[state_int][a1] += 1
                        self.value[state_int][a1] += cur_value
                        self.value_avg[state_int][a1] =\
                            self.value[state_int][a1] / self.visit_count[state_int][a1]
                cur_value = -cur_value

    def get_policy_value(self, state_int, movel, cur_player, tau=1):
        """
        Extract policy and action-values by the state
        :param state_int: state of the board
        :return: (probs, values)
        """
        counts = self.visit_count[state_int]
        movep = []
        for m in movel:
            if m<1: movep.append(counts[90])
            elif m>9999: movep.append(counts[91+m-10000])
            else:
                if cur_player>0: m = hanAction(m)
                movep.append(counts[m//100]+counts[m%100+95])
        total = sum(movep)
        if tau == 0 or total<1:
            probs = [0.0] * model.policy_size
            a = random.randrange(0, len(movel)) if total<1 else np.argmax(movep)
            m = movel[a]
            if m<1: probs[90] = 1
            elif m>9999: probs[91+m-10000] = 1
            else:
                if cur_player > 0: m = hanAction(m)
                probs[m//100] =  probs[m%100+95] = 0.5
            movep = [0.0] * len(movel)
            movep[a] = 1
        else:
            for i,m in enumerate(movep): movep[i] = m/total
            total = sum(counts)
            probs = [count / total for count in counts]

        values = self.value_avg[state_int]
        return probs, values, movep
