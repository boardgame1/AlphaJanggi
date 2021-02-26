import collections, json, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib import game, mcts, webFunction, actionTable


NUM_FILTERS = 512
OBS_SHAPE = (1, game.GAME_ROWS, game.GAME_COLS)

class Net(nn.Module):
    def __init__(self, actions_n):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, NUM_FILTERS, 3, padding=1)
        self.conv2 = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, 3, padding=1)
        self.convA = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, 3)
        self.convB = nn.Conv2d(NUM_FILTERS, NUM_FILTERS, 3)

        self.bn1 = nn.BatchNorm2d(NUM_FILTERS)
        self.bn2 = nn.BatchNorm2d(NUM_FILTERS)
        self.bnA = nn.BatchNorm2d(NUM_FILTERS)
        self.bnB = nn.BatchNorm2d(NUM_FILTERS)

        self.fc1 = nn.Linear(NUM_FILTERS * (game.GAME_ROWS - 4) * (game.GAME_COLS - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, actions_n)

        self.fc4 = nn.Linear(512, 1)

        self.dropout = 0.3

    def train(self, mode: bool = True):
        super(Net, self).train(mode)
        self.training = mode

    def eval(self):
        super(Net, self).eval()
        self.training = False

    def forward(self, s):
        s = s.view(-1, 1, game.GAME_ROWS, game.GAME_COLS)            # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bnA(self.convA(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bnB(self.convB(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, NUM_FILTERS*(game.GAME_ROWS-4)*(game.GAME_COLS-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return pi, torch.tanh(v)


def _encode_list_state(dest_np, state_list, step):
    assert dest_np.shape == OBS_SHAPE

    who_move = step %2
    for row_idx, row in enumerate(state_list):
        for col_idx, cell in enumerate(row):
            if cell>0:
                dest_np[0, row_idx, col_idx] = cell%10*((cell//10 if who_move<1 else 1-cell//10)*2-1)

def state_lists_to_batch(state_lists, steps_lists, device="cpu"):
    assert isinstance(state_lists, list)
    batch_size = len(state_lists)
    batch = np.zeros((batch_size,) + OBS_SHAPE, dtype=np.float32)
    for idx, (state, step) in enumerate(zip(state_lists, steps_lists)):
        _encode_list_state(batch[idx], state, step)
    return torch.tensor(batch).to(device)


def play_game(value, mcts_stores, queue, net1, net2, steps_before_tau_0, mcts_searches, mcts_batch_size,
              best_idx, url=None, username=None, device="cpu"):
    assert isinstance(mcts_stores, (mcts.MCTS, type(None), list))
    assert isinstance(net1, Net)
    assert isinstance(net2, Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(), mcts.MCTS()]
    elif isinstance(mcts_stores, mcts.MCTS):
        mcts_stores = [mcts_stores, mcts_stores]

    state = game.encode_lists([list(i) for i in game.INITIAL_STATE], 0)
    nets = [net1, net2]
    cur_player = 0
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    net1_result = None
    result = None

    while net1_result is None and (value==None or value[0]>0):
        mcts_stores[cur_player].clear()
        mcts_stores[cur_player].search_batch(mcts_searches, mcts_batch_size, state,
                                             cur_player, nets[cur_player], step, device=device)
        movel = game.possible_moves(state, cur_player, step)
        probs, _ = mcts_stores[cur_player].get_policy_value(state, movel, tau=tau)
        action = actionTable.moveTable[np.random.choice(actionTable.AllMoveLength, p=probs)]
        game_history.append((action, probs) if queue is None else (state, step, probs))
        if action not in movel:
            print("Impossible action selected")
        state, won = game.move(state, action, step)
        if (best_idx>=0 or value==None) and step%3<1: print('.', end='', flush=True)
        if won>0:
            net1_result = 1 if won == 1 else -1
            result = -net1_result
            break
        step += 1
        cur_player = 1-cur_player
        if step >= steps_before_tau_0:
            tau = 0

    if net1_result !=None:
        if best_idx>=0 or value==None: print()
        if queue is not None:
            dequeuef = isinstance(queue, collections.deque)
            for state, hstep, probs in game_history:
                queue.append((state, hstep, probs, result)) if dequeuef else\
                    queue.put((state, hstep, probs, result))
                if hstep!=1: result = -result
        elif best_idx>=0:
            gh = []
            for (action, probs) in game_history:
                prar = []
                for idx, prob in enumerate(probs):
                    if prob>0: prar.append([idx, prob])
                gh.append((action, prar))
            js = {"netIdx":best_idx, "result":net1_result, "username":username, "action":gh}
            hr = webFunction.http_request(url, True, json.dumps(js))
            if hr == None: sys.exit()
            elif hr['status'] == 'error': print('error occured')
            else: print("game is uploaded")

    return net1_result, step if net1_result!=None else 0
