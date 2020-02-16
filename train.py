#!/usr/bin/env python3
import os
import json
import random
import copy
import argparse
import collections

from lib import game, model, mcts

import torch
import torch.optim as optim
import torch.nn.functional as F


PLAY_EPISODES = 25
REPLAY_BUFFER = 30000
LEARNING_RATE = 0.01
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 10000

BEST_NET_WIN_RATIO = 0.55

EVALUATION_ROUNDS = 20

def evaluate(net1, net2, rounds, device="cpu"):
    n1_win, n2_win = 0, 0
    mcts_stores = [mcts.MCTS(), mcts.MCTS()]

    for r_idx in range(rounds):
        r, step = model.play_game(None, mcts_stores, None, net1 if r_idx<rounds//2 else net2,
                    net2 if r_idx<rounds//2 else net1, steps_before_tau_0=game.MAX_TURN, mcts_searches=20,
                    mcts_batch_size=20, best_idx=-1, device=device)
        if (r > 0 and r_idx<rounds//2) or (r < 0 and r_idx>=rounds//2):
            n1_win += 1
        if r!=0: n2_win += 1
        print(r_idx, r, step)
    return (n1_win / n2_win) if n2_win>0 else 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("-m", "--model", help="Model to load")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    saves_path = "saves"
    os.makedirs(saves_path, exist_ok=True)

    net = model.Net(input_shape=model.OBS_SHAPE, actions_n=model.policy_size).to(device)
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    step_idx = 0
    best_idx = 1

    if args.model:
        checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['model'], strict=False)
        best_idx = checkpoint['best_idx']
        print("model loaded", args.model)
    best_net = copy.deepcopy(net)
    best_net.eval()
    print('best_idx: '+str(best_idx))

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    f = open("./train.dat", "r")

    while True:
        for lidx in range(PLAY_EPISODES):
            pan = game.encode_lists([list(i) for i in game.INITIAL_STATE], 0)

            s = f.readline()
            if len(s)<5: lidx -= 1; break
            js = json.loads(s)
            result = -js["result"]
            for idx, (action, probs) in enumerate(js["action"]):
                movelist, _ = game.possible_moves(pan, idx%2, idx)
                if action not in movelist:
                    print("Impossible action selected "+step_idx+" "+lidx)
                probs1 = [0.0] * model.policy_size
                for n in probs:
                    probs1[n[0]] = n[1]
                replay_buffer.append((pan, idx, probs1, result))
                pan, _ = game.move(pan, action, idx)
                if idx!=1: result = -result
        if lidx < 0: break

        print(step_idx)
        step_idx += 1
        if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
            continue

        print("train %d" % (step_idx))

        net.train()
        for _ in range(TRAIN_ROUNDS):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            batch_states, batch_steps, batch_probs, batch_values = zip(*batch)
            batch_states_lists = [game.decode_binary(state) for state in batch_states]
            states_v = model.state_lists_to_batch(batch_states_lists, batch_steps, device)

            optimizer.zero_grad()
            probs_v = torch.FloatTensor(batch_probs).to(device)
            values_v = torch.FloatTensor(batch_values).to(device)
            out_logits_v, out_values_v = net(states_v)

            loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
            loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
            loss_policy_v = loss_policy_v.sum(dim=1).mean()

            loss_v = loss_policy_v + loss_value_v
            loss_v.backward()
            optimizer.step()
    f.close()

    print("Net evaluation started")
    net.eval()
    win_ratio = evaluate(net, best_net, rounds=EVALUATION_ROUNDS, device=device)
    print("Net evaluated, win ratio = %.2f" % win_ratio)
    if win_ratio >= BEST_NET_WIN_RATIO:
        print("Net is better than cur best, sync")
        best_net.load_state_dict(net.state_dict())
        best_idx += 1
        file_name = os.path.join(saves_path, "best_%d.pth" % (best_idx))
        torch.save({'model': net.state_dict(), 'best_idx': best_idx}, file_name)

