# coding=utf-8
import os, sys, getpass
import time
import argparse, json

from lib import model, mcts, webFunction, game, actionTable

import torch

MCTS_SEARCHES = 10
MCTS_BATCH_SIZE = 20
STEPS_BEFORE_TAU_0 = 20
domain = "https://alphajanggi.net"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

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
        hr = webFunction.http_request(domain+"/user7", True, json.dumps(js))
        if hr == None:
            print("문제가 지속되면 프로젝트 사이트에서 프로그램을 다시 다운로드하세요.")
            sys.exit()
        if hr["status"] == "ok": break
        if hr["status"] == "dup":
            print("duplicate user ID"); continue
        elif hr["status"] == "notexist":
            print("user ID does not exist or password is incorrect"); continue

    best_idx = 0
    mhash = 1
    dfile = "./selfmodel.dat"
    modelfile = "./best_model.pth"
    if os.path.isfile(dfile):
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

    step_idx = 0

    checkpoint = torch.load(modelfile, map_location=lambda storage, loc: storage);
    if(checkpoint['best_idx'] != best_idx):
        print("wrong model file")
        sys.exit()
    if 'resBlockNum' in checkpoint: model.resBlockNum = checkpoint['resBlockNum']
    net = model.Net(input_shape=model.OBS_SHAPE, actions_n=actionTable.AllMoveLength).to(device)
    net.load_state_dict(checkpoint['model'], strict=False)
    net.eval()

    mcts_store = mcts.MCTS()

    while True:
        t = time.time()
        _, game_steps = model.play_game(None, mcts_store, None, net, net, steps_before_tau_0=game.MAX_TURN,
                mcts_searches=MCTS_SEARCHES, mcts_batch_size=MCTS_BATCH_SIZE, best_idx=best_idx,
                                   domain=domain, username=username, device=device)
        game_nodes = len(mcts_store)
        dt = time.time() - t
        speed_steps = game_steps / dt
        speed_nodes = game_nodes / dt
        print("Step %d, steps %3d, leaves %4d, steps/s %5.2f, leaves/s %6.2f, best_idx %d" % (
            step_idx, game_steps, game_nodes, speed_steps, speed_nodes, best_idx))
        step_idx += 1
