import collections, json, sys
import numpy as np

import torch
import torch.nn as nn

from lib import game, mcts, webFunction, actionTable


NUM_FILTERS = 128
OBS_SHAPE = (17, game.GAME_ROWS, game.GAME_COLS)
resBlockNum = 1

class Net(nn.Module):
    def __init__(self, input_shape, actions_n):
        super(Net, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS), nn.LeakyReLU())

        res_block = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS), nn.LeakyReLU())

        self.blocks = nn.ModuleList([res_block for _ in range(resBlockNum)])

        body_out_shape = (NUM_FILTERS, ) + input_shape[1:]

        self.conv_val = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        conv_val_size = self._get_conv_val_size(body_out_shape)
        self.value = nn.Sequential(
            nn.Linear(conv_val_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )

        self.conv_policy = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_policy_size, actions_n)
        )

    def _get_conv_val_size(self, shape):
        o = self.conv_val(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _get_conv_policy_size(self, shape):
        o = self.conv_policy(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        v = self.conv_in(x)
        for block in self.blocks:
            v = v + block(v)
        val = self.conv_val(v)
        val = self.value(val.view(batch_size, -1))
        pol = self.conv_policy(v)
        pol = self.policy(pol.view(batch_size, -1))
        return pol, val


def _encode_list_state(dest_np, state_list, step):
    assert dest_np.shape == OBS_SHAPE

    who_move = step %2
    for row_idx, row in enumerate(state_list):
        ridx = row_idx if who_move < 1 else 9 - row_idx
        for col_idx, cell in enumerate(row):
            if cell>0:
                dest_np[cell%10-1+(cell//10 if who_move<1 else 1-cell//10)*7, ridx, col_idx] = 1.0
    ci=8
    while step>0:
        if step%2>0: dest_np[14, 0, ci] = 1.0
        step //=2
        ci -= 1

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
        mcts_stores.clear()
        mcts_stores = [mcts_stores, mcts_stores]
    else: mcts_stores[0].clear(); mcts_stores[1].clear()

    state = game.encode_lists([list(i) for i in game.INITIAL_STATE], 0)
    nets = [net1, net2]
    cur_player = 0
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    net1_result = None
    result = None

    while net1_result is None and (value==None or value[0]>0):
        mcts_stores[cur_player].search_batch(mcts_searches, mcts_batch_size, state,
                                             cur_player, nets[cur_player], step, device=device)
        movel = game.possible_moves(state, cur_player, step)
        probs, _ = mcts_stores[cur_player].get_policy_value(state, movel, cur_player, tau=tau)
        chList = actionTable.choList if cur_player < 1 else actionTable.hanList
        action = chList[np.random.choice(actionTable.AllMoveLength, p=probs)]
        game_history.append((action, probs) if queue is None else (state, step, probs))
        if action not in movel:
            print("Impossible action selected")
        state, won = game.move(state, action, step)
        if step%3<1: print('.', end='', flush=True)
        if won>0:
            net1_result = 1 if won == 1 else -1
            result = -net1_result
            break
        step += 1
        cur_player = 1-cur_player
        if step >= steps_before_tau_0:
            tau = 0

    if net1_result !=None:
        print()
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
