#!/usr/bin/env python3
import os, sys
import json
import random
import copy
import argparse
import collections, time

from lib import game, model, mcts, actionTable

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp


PLAY_EPISODES = 25
REPLAY_BUFFER = 30000
LEARNING_RATE = 0.1
BATCH_SIZE = 256
TRAIN_ROUNDS = 20
MIN_REPLAY_TO_TRAIN = 10000

BEST_NET_WIN_RATIO = 0.545
NUM_PROC = 5
EVALUATION_ROUNDS = 100

def eval(val, lock, net1, net2, device, cpuf):
    if cpuf: net1.to(device); net2.to(device)
    mcts_stores = [mcts.MCTS(), mcts.MCTS()]
    while True:
        are = random.randrange(0, 2)
        r, _ = model.play_game(val, mcts_stores, None, net1=net1 if are<1 else net2,
                net2=net2 if are<1 else net1, steps_before_tau_0=0,
                            mcts_searches=40, mcts_batch_size=40, best_idx=-1, device=device)

        bf = False
        lock.acquire()
        if r!=None:
            val[1 if (r > 0.5 and are<1) or (r<-0.5 and are>=1) else 2] += 1
            print("%d:%d %d/%d"%(are,r,val[1],val[2]),end=' ', flush=True)
            if (val[1]+val[2]) % 5 <1: print()
        if val[0]<=0: bf=True
        lock.release()
        if bf: break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("-m", "--model", help="Model to load")
    parser.add_argument("-tm", "--tmodel", help="Temp model")
    args = parser.parse_args()
    device = torch.device("cuda:1" if args.cuda else "cpu")

    saves_path = "saves"
    os.makedirs(saves_path, exist_ok=True)

    step_idx = 0

    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
    best_net = model.Net(input_shape=model.OBS_SHAPE, actions_n=actionTable.AllMoveLength).to(device)
    best_net.load_state_dict(checkpoint['model'], strict=False)
    best_idx = checkpoint['best_idx']
    #print(best_net)
    if args.tmodel:
        checkpoint = torch.load(args.tmodel, map_location=lambda storage, loc: storage)
        if best_idx != checkpoint['best_idx']: print('invalid tmodel'); sys.exit()
        net = model.Net(input_shape=model.OBS_SHAPE, actions_n=actionTable.AllMoveLength).to(device)
        net.load_state_dict(checkpoint['model'])
    else: net = copy.deepcopy(best_net)
    best_net.eval()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)
    if args.tmodel: optimizer.load_state_dict(checkpoint['opt'])
    print('best_idx: '+str(best_idx))

    net.train()
    replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
    f = open("./train.dat", "r")
    ptime = time.time()

    while True:
        for lidx in range(PLAY_EPISODES):
            pan = game.encode_lists([list(i) for i in game.INITIAL_STATE], 0)

            s = f.readline()
            if len(s)<5: lidx -= 1; break
            js = json.loads(s)
            result = -js["result"]
            for idx, (action, probs) in enumerate(js["action"]):
                movelist = game.possible_moves(pan, idx%2, idx)
                #if action not in movelist:
                #    print("Impossible action selected %d %d"%(step_idx, lidx))
                probs1 = [0.0] * actionTable.AllMoveLength
                for n in probs:
                    probs1[n[0]] = n[1]
                replay_buffer.append((pan, idx, probs1, result))
                pan, _ = game.move(pan, action, idx)
                if idx!=1: result = -result
        if lidx < 0: break

        print(step_idx, end=' ')
        step_idx += 1
        if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
            continue

        ctime=time.time()
        print("%.2f "%(ctime-ptime), end=' ')
        if step_idx%5<1: print()
        ptime=ctime

        for _ in range(TRAIN_ROUNDS):
            batch = random.sample(replay_buffer, BATCH_SIZE)
            batch_states, batch_steps, batch_probs, batch_values = zip(*batch)
            batch_states_lists = [game.decode_binary(state) for state in batch_states]
            states_v = model.state_lists_to_batch(batch_states_lists, batch_steps, device)

            optimizer.zero_grad()
            probs_v = torch.FloatTensor(batch_probs).to(device)
            values_v = torch.FloatTensor(batch_values).to(device)
            out_logits_v, out_values_v = net(states_v)
            #traced_script_module = torch.jit.trace(net, states_v)
            #file_name = os.path.join(saves_path, "best_%d.pt" % (best_idx))
            #traced_script_module.save(file_name); sys.exit()
            loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
            loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
            loss_policy_v = loss_policy_v.sum(dim=1).mean()

            loss_v = loss_policy_v + loss_value_v
            loss_v.backward()
            optimizer.step()
    f.close()

    if args.tmodel==None or args.tmodel.find('_a.')<0:
        cn=0
        if args.tmodel:
            f=open('./count.txt', 'r'); s=f.readline(); cn=int(s); f.close()
        f = open('./count.txt', 'w'); cn+=step_idx; f.write(str(cn)+'\n'); f.close()
    fns = args.tmodel if args.tmodel else "best_%d.pth" % (best_idx)
    file_name = os.path.join('.', fns)
    torch.save({'model': net.state_dict(), 'best_idx': best_idx,
                'opt': optimizer.state_dict()}, file_name)

    print("Net evaluation started")
    net.eval()
    if os.name == 'nt' and args.cuda:
        cd = torch.device("cpu")
        net.to(cd)
        best_net.to(cd)
        cpuf = True
    else: cpuf = False

    mp.set_start_method("spawn", force=True)
    lock = mp.Lock()
    processes = [];
    mar = mp.Array('i', 3);
    mar[0] = 1
    for i in range(NUM_PROC):
        p = mp.Process(target=eval, args=(mar, lock, net, best_net, device, cpuf), daemon=True)
        p.start()
        processes.append(p)
    while 1:
        lock.acquire()
        if mar[0] > 0 and (mar[1] >= EVALUATION_ROUNDS*BEST_NET_WIN_RATIO or
                           mar[2]>EVALUATION_ROUNDS*(1-BEST_NET_WIN_RATIO)):
            mar[0] = 0
        lock.release()
        running = any(p.is_alive() for p in processes)
        if not running:
            break
        time.sleep(0.5)
    print()

    if mar[1] >= EVALUATION_ROUNDS*BEST_NET_WIN_RATIO:
        print("Net is better than cur best, sync")
        best_idx += 1
        file_name = os.path.join(saves_path, "best_%d.pth" % (best_idx))
        torch.save({'model': net.state_dict(), 'best_idx': best_idx}, file_name)
