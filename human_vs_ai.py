# coding=utf-8
import argparse, os
import numpy as np
import torch
from lib import game, mcts, model

piece_str = u"초차포마상사졸漢車包馬象士兵"
def render(pan_str, player_human):
    pan = game.decode_binary(pan_str)
    print("   1  2  3  4  5  6  7  8  9")
    for y in range(10):
        s = chr((10-y if y>0 else 0)+ord('0'))+" "
        for x in range(9):
            a = pan[y if player_human>0 else 9-y][x if player_human>0 else 8-x]
            s += piece_str[a // 10 * 7 + a % 10 - 1] + ('-' if x < 8 else ' ') if a > 0 else \
                (('┌' if x < 1 else '┬' if x < 8 else '┐') if y < 1 else \
                ('├' if x < 1 else '╋' if x < 8 else '┤') if y < 9 else \
                ('└' if x < 1 else '┴' if x < 8 else '┘')) + ('─-' if x < 8 else ' ')
        print(s)
        if y<9:
            print("  │  │  │  │＼│／│  │  │  │ " if y<1 or y==7 else "  │  │  │  │／│＼│  │  │  │ "\
                if y==1 or y>7 else "  │  │  │  │  │  │  │  │  │ ")

def play_game(net1, steps_before_tau_0, mcts_searches, mcts_batch_size, device="cpu"):
    assert isinstance(net1, model.Net)
    assert isinstance(steps_before_tau_0, int) and steps_before_tau_0 >= 0
    assert isinstance(mcts_searches, int) and mcts_searches > 0
    assert isinstance(mcts_batch_size, int) and mcts_batch_size > 0

    pan = game.encode_lists([list(i) for i in game.INITIAL_STATE], 0)
    capturelist = []
    actionhistory = []
    cur_player = 0
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    mctsi = mcts.MCTS()

    result = None
    a0 = ord('1')
    s=input('플레이하려는 진영을 선택하세요 0) 초, 1)한 ?')
    player_human = 0 if int(s)<1 else 1

    while result is None:
        movelist, _ = game.possible_moves(pan, cur_player, step)
        if (step<2 and cur_player != player_human) or (step>1 and cur_player == player_human):
            if step < 2:
                print("마상 차림을 선택하세요 0) 마상마상, 1) 상마상마, 2) 마상상마, 3) 상마마상")
            else:
                render(pan, player_human)
                if step==2 or step==3:
                    print("")
                    print("옮기고자 하는 기물의 가로 번호, 세로 번호, 목적지의 가로 번호, 세로 번호 ex) 0010  한수쉼: 0")
            action = -1
            while action<0:
                s=input((str(step-1) if step>1 else '')+' ? ')
                if step<2:
                    if len(s)==1 and s[0]>='0' and s[0]<'4': action = int(s) + 10000
                elif len(s)==1: action = 0
                elif len(s)==4 and s[0]>='0' and s[0]<='9' and s[1]>'0' and s[1]<='9' and s[2]>='0' and s[2]<='9' and s[3]>'0' and s[3]<='9':
                    b1=9-ord(s[0])+a0 if s[0]>'0' else 0
                    if player_human<1: b1=9-b1
                    b2 = ord(s[1]) - a0
                    if player_human < 1: b2 = 8 - b2
                    b3 = 9-ord(s[2]) + a0 if s[2]>'0' else 0
                    if player_human < 1: b3 = 9 - b3
                    b4 = ord(s[3]) - a0
                    if player_human < 1: b4 = 8 - b4
                    action = (b1*9 + b2)*100 + b3*9+b4
                if action not in movelist: action = -1
        else:
            mctsi.search_batch(mcts_searches, mcts_batch_size, pan,
                            cur_player, net1, step, device=device)
            probs, _, movep = mctsi.get_policy_value(pan, movelist, cur_player, tau=tau)
            action = movelist[np.random.choice(len(movelist), p=movep)]
            if step<1: print('한: '+('마상마상' if action<10001 else '상마상마' if action<10002 else '마상상마' if action<10003 else '상마마상'))
            elif step>1:
                if action<1: print('한수쉼')
                else:
                    b1=action//100//9
                    if player_human<1: b1=9-b1
                    b2 = action//100%9
                    if player_human < 1: b2 = 8 - b2
                    b3 = action%100//9
                    if player_human < 1: b3 = 9 - b3
                    b4 = action%100%9
                    if player_human < 1: b4 = 8 - b4
                    print((chr(9-b1+a0) if b1>0 else '0')+chr(b2+a0)+(chr(9-b3+a0) if b3>0 else '0')+chr(b4+a0))
        pan, won = game.move(pan, action, step)
        actionhistory.append(action)
        if won>0:
            render(pan, player_human)
            print(('초' if won==1 else '한')+' 승')
            break
        cur_player = 1-cur_player
        step += 1
        if step >= game.MAX_TURN:
            tau = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="The model to play")
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable CUDA")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    modelfile = args.model if args.model else "./best_model.pth"
    if os.path.isfile(modelfile):
        net = model.Net(model.OBS_SHAPE, model.policy_size).to(device)
        checkpoint = torch.load(modelfile, map_location=lambda storage, loc: storage)
        net.load_state_dict(checkpoint['model'], strict=False)
        net.eval()

        play_game(net, 20, 60, 80, device)
    else:
        print(modelfile+" 파일이 존재하지 않습니다")
