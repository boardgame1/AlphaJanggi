# coding=utf-8
import os, sys, getpass
import time
import argparse, json

from lib import model, mcts, webFunction, actionTable

import torch
import torch.multiprocessing as mp

MCTS_SEARCHES = 20
MCTS_BATCH_SIZE = 40
STEPS_BEFORE_TAU_0 = 20
PLAY_EPISODE = 30
domain = "https://alphajanggi.net"

def play(val, lock, mcts_store, net, best_idx, username, device, step_idx):
    while True:
        t = time.time()
        _, game_steps = model.play_game(None, mcts_store, None, net, net, steps_before_tau_0=STEPS_BEFORE_TAU_0,
                            mcts_searches=MCTS_SEARCHES, mcts_batch_size=MCTS_BATCH_SIZE, best_idx=best_idx,
                            url=domain + "/selfplay10", username=username, device=device)
        game_nodes = len(mcts_store)
        dt = time.time() - t
        speed_steps = game_steps / dt
        speed_nodes = game_nodes / dt

        bf = False
        lock.acquire()
        val[1] += 1
        print("Step %d, steps %3d, leaves %4d, steps/s %5.2f, leaves/s %6.2f, best_idx %d" % (
            step_idx+val[1], game_steps, game_nodes, speed_steps, speed_nodes, best_idx))
        if val[0]>0: val[0] -= 1
        else: bf=True
        lock.release()
        if bf: break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    parser.add_argument("--numproc", type=int, default=1, help="Number of process")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    num_proc = args.numproc

    username = ""
    while True:
        createf = False
        username = input("user ID (to create, enter 0): ")
        if username == "": continue
        if username == '0':
            username = input("user ID to create: ")
            if username == "": continue
            password = getpass.getpass("password: ")
            password2 = getpass.getpass("password to confirm: ")
            if password == "" or password!=password2: continue
            createf = True
        else:
            password = getpass.getpass("password: ")
            if password == "": continue
        js = {"username":username, "password":password, "createf":createf}
        hr = webFunction.http_request(domain+"/user10", True, json.dumps(js))
        if hr == None:
            print("문제가 지속되면 프로젝트 사이트에서 프로그램을 다시 다운로드하세요.")
            sys.exit()
        if hr["status"] == "ok": break
        if hr["status"] == "dup":
            print("duplicate user ID"); continue
        elif hr["status"] == "notexist":
            print("user ID does not exist or password is incorrect"); continue

    step_idx = 0
    mp.set_start_method("spawn", force=True)
    lock = mp.Lock()

    while True:
        print('checking model')
        best_idx = 0
        mhash = -1
        dfile = "./selfmodel.dat"
        modelfile = "./best_model.pth"
        if os.path.isfile(dfile) and os.path.isfile(modelfile):
            df = open(dfile, "r"); s = df.readline(); df.close()
            mhash = int(s[:8], 16)
            best_idx = int(s[8:], 16)
        s = domain+"/checkmodel"
        if mhash >=0: s += "?mhash="+str(mhash)
        hr = webFunction.http_request(s)
        if hr == None: sys.exit()
        if hr["status"] == "download":
            webFunction.download_file(domain+hr["url"], modelfile)
            df = open(dfile, "w")
            s = '{:08x}'.format(hr["hash"]) + '{:x}'.format(hr["idx"])
            best_idx = hr["idx"]
            df.write(s)
            df.close()

        checkpoint = torch.load(modelfile, map_location=lambda storage, loc: storage);
        if(checkpoint['best_idx'] != best_idx):
            print("wrong model file")
            sys.exit()
        if 'resBlockNum' in checkpoint: model.resBlockNum = checkpoint['resBlockNum']
        net = model.Net(input_shape=model.OBS_SHAPE, actions_n=actionTable.AllMoveLength).to(device)
        net.load_state_dict(checkpoint['model'], strict=False)
        net.eval()
        net.share_memory()

        if os.name == 'nt' and args.cuda:
            mcts_store = mcts.MCTS()
            for i in range(PLAY_EPISODE):
                t = time.time()
                _, game_steps = model.play_game(None, mcts_store, None, net, net,
                    steps_before_tau_0=STEPS_BEFORE_TAU_0, mcts_searches=MCTS_SEARCHES,
                    mcts_batch_size=MCTS_BATCH_SIZE, best_idx=best_idx,
                    url=domain+"/selfplay10", username=username, device=device)
                game_nodes = len(mcts_store)
                dt = time.time() - t
                speed_steps = game_steps / dt
                speed_nodes = game_nodes / dt
                step_idx += 1
                print("Step %d, steps %3d, leaves %4d, steps/s %5.2f, leaves/s %6.2f, best_idx %d" % (
                    step_idx, game_steps, game_nodes, speed_steps, speed_nodes, best_idx))
        else:
            processes = []; mar = mp.Array('i', 2); mar[0] = PLAY_EPISODE * num_proc
            for i in range(num_proc):
                mcts_store = mcts.MCTS()
                p = mp.Process(target=play, args=(mar, lock, mcts_store, net, best_idx,
                                                  username, device, step_idx), daemon=True)
                p.start()
                processes.append(p)
            while 1:
                running = any(p.is_alive() for p in processes)
                if not running:
                    break
                time.sleep(0.5)
            step_idx += mar[1]
            print()
