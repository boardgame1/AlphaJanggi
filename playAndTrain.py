#!/usr/bin/env python3
import os, copy
import time
import random
import argparse
import collections

from lib import game, model, mcts, webFunction

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp

PLAY_EPISODES = 25
MCTS_SEARCHES = 10
MCTS_BATCH_SIZE = 20
REPLAY_BUFFER = 30000
LEARNING_RATE = 0.1
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 10000

BEST_NET_WIN_RATIO = 0.55

EVALUATE_EVERY_STEP = 100
EVALUATION_ROUNDS = 20
STEPS_BEFORE_TAU_0 = 20

os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

def eval(lock, mar, net1, net2, mcts_searches, mcts_batch_size, device):
    mcts_stores = [mcts.MCTS(), mcts.MCTS()]
    while True:
        are = random.randrange(0, 2)
        r, _ = model.play_game(mar, mcts_stores=mcts_stores, queue=None, net1=net1 if are<1 else net2,
                net2=net2 if are<1 else net1, steps_before_tau_0=0, mcts_searches=mcts_searches,
                               mcts_batch_size=mcts_batch_size, device=device);
        if r!=None:
            print(are, r)
            if (r > 0.5 and are<1) or (r<-0.5 and are>0):
                mar[1] += 1
            if r!=0: mar[2]+=1
        bf = False
        lock.acquire()
        if mar[0]>0: mar[0] -= 1
        else: bf=True
        lock.release()
        if bf: break

def evaluate(numproc, cudaf, lock, net1, net2, rounds, mcts_searches, mcts_batch_size, device="cpu"):
    if os.name != 'nt' or cudaf!=True:
        processes = []
        mar = mp.Array('i', 3); mar[0]=rounds
        for i in range(numproc):
            p = mp.Process(target=eval, args=(lock, mar, net1, net2, mcts_searches, mcts_batch_size, device), daemon=True)
            p.start(); processes.append(p)
        for p in processes: p.join()
        return mar[1] / mar[2] if mar[2]>0 else 0.5
    else:
        n1_win, n2_win = 0, 0
        mcts_stores = [mcts.MCTS(), mcts.MCTS()]
        for k in range(2):
            for r_idx in range(rounds//2):
                r, _ = model.play_game(None, mcts_stores, queue=None, net1=net1 if k<1 else net2,
                        net2=net2 if k<1 else net1, steps_before_tau_0=0, mcts_searches=mcts_searches,
                                       mcts_batch_size=mcts_batch_size, device=device); print(k, r)
                if r != 0:
                    n2_win += 1
                if (r > 0.5 and k<1) or (r<-0.5 and k>0):
                    n1_win += 1
        return n1_win / n2_win if n2_win>0 else 0.5

def play(val, lock, mcts_store, replay_buffer, net1, net2, steps_before_tau_0,
         mcts_searches, mcts_batch_size, device="cpu"):
    while True:
        _, steps = model.play_game(val, mcts_store, replay_buffer, net1, net2,
                                   steps_before_tau_0, mcts_searches, mcts_batch_size, device)
        val[1] += steps
        bf = False
        lock.acquire()
        if val[0]>0: val[0] -= 1
        else: bf=True
        lock.release()
        if bf: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--numproc", type=int, default=1, help="Number of process")
    parser.add_argument("-m", "--model", help="Model to load")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    num_proc = args.numproc

    saves_path = "saves"
    os.makedirs(saves_path, exist_ok=True)

    best_idx = 0
    net = model.Net(input_shape=model.OBS_SHAPE, actions_n=model.policy_size).to(device)
    #print(net)
    if args.model: modelfile = args.model
    else:
        modelfile = './model.pth'
        webFunction.download_file('https://alphajanggi.net/modeldownload2', modelfile)

    checkpoint = torch.load(modelfile, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['model'], strict=False)
    best_idx = checkpoint['best_idx']
    print("model loaded", modelfile, best_idx)

    best_net = copy.deepcopy(net)
    best_net.eval(); net.train()

    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    step_idx = 0

    best_net.share_memory()
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue();  lock = mp.Lock()
    while True:
        t = time.time()
        #prev_nodes = len(mcts_store)
        game_steps = 0
        if os.name == 'nt' and args.cuda:
            mcts_store = mcts.MCTS()
            for _ in range(PLAY_EPISODES):
                _, steps = model.play_game(None, mcts_store, replay_buffer, best_net, best_net,
                                       STEPS_BEFORE_TAU_0, MCTS_SEARCHES, MCTS_BATCH_SIZE, device)
                game_steps += steps
        else:
            processes = []; mar = mp.Array('i', 2); mar[0] = PLAY_EPISODES
            for i in range(num_proc):
                mcts_store = mcts.MCTS();
                p = mp.Process(target=play, args=(mar, lock, mcts_store, queue, best_net, best_net,
                        STEPS_BEFORE_TAU_0, MCTS_SEARCHES, MCTS_BATCH_SIZE, device), daemon=True)
                p.start()
                processes.append(p)
            while 1:
                running = any(p.is_alive() for p in processes)
                while not queue.empty():
                    replay_buffer.append(queue.get(False))
                if not running:
                    break
                time.sleep(0.5)
            game_steps += mar[1]
            print()

        game_nodes = len(mcts_store)# - prev_nodes
        dt = time.time() - t
        speed_steps = game_steps / dt
        print("Step %d, steps %3d, leaves %4d, steps/s %5.2f, dt %6.2f, best_idx %d, replay %d" % (
            step_idx, game_steps, game_nodes, speed_steps, dt, best_idx, len(replay_buffer)))
        step_idx += 1

        if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
            continue

        # train
        sum_loss = 0.0
        sum_value_loss = 0.0
        sum_policy_loss = 0.0

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
            sum_loss += loss_v.item()
            sum_value_loss += loss_value_v.item()
            sum_policy_loss += loss_policy_v.item()

        # evaluate net
        if step_idx % EVALUATE_EVERY_STEP == 0:
            net.eval()
            win_ratio = evaluate(num_proc, args.cuda, lock, net, best_net, rounds=EVALUATION_ROUNDS,
                                 mcts_searches=20, mcts_batch_size=20, device=device)
            print("Net evaluated, win ratio = %.2f" % win_ratio)
            if win_ratio > BEST_NET_WIN_RATIO:
                print("Net is better than cur best, sync")
                best_net.load_state_dict(net.state_dict())
                best_idx += 1
                file_name = os.path.join(saves_path, "best_%d.pth" % (best_idx))
                torch.save({'model': net.state_dict(), 'best_idx': best_idx}, file_name)

                a1 = file_name.rfind('\\')
                a2 = file_name.rfind('/')
                if a2 > a1: a1 = a2
                webFunction.upload_file('https://alphajanggi.net/fileupload2', path=file_name,
                                        filename=file_name[(a1 if a1 > 0 else 0):])
            net.train()
