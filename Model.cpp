#include "pch.h"

void _encode_list_state(float dest_np[][10][9], int state_list[][9], int step) {
    int who_move = step % 2;
    for (int row_idx=0;row_idx<10;row_idx++)
        for (int col_idx = 0; col_idx < 9; col_idx++) {
            int cell = state_list[row_idx][col_idx];
            if (cell > 0)
                dest_np[cell % 10 - 1 + (who_move < 1 ? cell / 10 : 1 - cell / 10) * 7][row_idx][col_idx] = 1.0f;
        }
    int ci = 8;
    while (step > 0) {
        if (step % 2 > 0) dest_np[14][0][ci] = 1.0f;
        step /= 2;
        ci -= 1;
    }
    if (who_move > 0) dest_np[14][9][0] = 1.0f;
}

torch::jit::IValue state_lists_to_batch(const vector<string>& state_lists, const vector<int>& steps_lists, torch::Device device) {
    const int batch_size = state_lists.size();
    float batch[BATCH_SIZE][15][10][9] = { 0, };
    for (int idx = 0; idx < batch_size; idx++) {
        auto state = state_lists[idx];
        int pan[10][9]; decode_binary(pan, state);
        int step = steps_lists[idx];
        _encode_list_state(batch[idx], pan, step);
    }
    torch::jit::IValue t = torch::from_blob(batch, { batch_size, 15,10,9 }).to(device);
    return t;
}

tuple<int, int> play_game(int* value, shared_ptr<MCTS> mcts, shared_ptr<MCTS> mcts2, torch::jit::script::Module* const net1,
    torch::jit::script::Module* const net2, int steps_before_tau_0, int const mcts_searches, int best_idx,
    string url, string username, torch::Device device, httplib::Client* http) {
    if (mcts == nullptr) {
        mcts = make_shared<MCTS>(); mcts2 = make_shared<MCTS>();
    }
    else if (mcts2 == nullptr)
        mcts2 = mcts;
    mcts->clear(); mcts2->clear();
    array<shared_ptr<MCTS>, 2> mcts_stores = { mcts,mcts2 };

    string state = encode_lists(pani, 0);
    vector<torch::jit::script::Module*> nets = { net1, net2 };
    int cur_player = 0;
    int step = 0;
    float tau = steps_before_tau_0 > 0 ? 1 : 0;
    vector<tuple<int, array<float, AllMoveLength>>> game_history;

    int net1_result = 9;

    while (net1_result > 5 && (value==nullptr || value[0]>0)) {
        mcts_stores[cur_player]->search_batch(mcts_searches, state,
            cur_player, nets[cur_player], step, device);
        vector<int> const movel = possible_moves(state, cur_player, step);
        array<float, AllMoveLength> probs, v; tie(probs, v) = mcts_stores[cur_player]->get_policy_value(state, movel, tau);
        float tt = 0, f = urd(rdgen); int n;
        for (int n0 : movel) {
            n = moveDict[n0];
            tt += probs[n];
            if (tt >= f) break;
        }
        int action = moveTable[n];
        game_history.emplace_back(make_tuple(action, probs));
        //if (find(movel.begin(), movel.end(), action) == movel.end())
        //    cout << "Impossible action selected" << endl;
        int won; tie(state, won) = move(state, action, step);
        if ((best_idx>=0 || value==nullptr) && step % 3 < 1) cout << ".";
        if (won > 0) {
            net1_result = won == 1 ? 1 : -1;
            break;
        }
        step++;
        cur_player = 1 - cur_player;
        if (step >= steps_before_tau_0)
            tau = 0;
    }

    if (net1_result < 5) {
        if (best_idx >= 0 || value == nullptr) cout << endl;
        if (best_idx >= 0) {
            vector<tuple<int, vector<tuple<int, float>>>> gh;
            for (int i = 0; i < game_history.size(); i++) {
                int action; array<float, AllMoveLength> probs;
                tie(action, probs) = game_history[i];
                vector<tuple<int, float>> prar;
                for (int idx = 0; idx < probs.size(); idx++) {
                    float prob = probs[idx];
                    if (prob > 0) prar.emplace_back(make_tuple(idx, prob));
                }
                gh.emplace_back(make_tuple(action, prar));
            }
            json js;
            js["action"] = gh; js["netIdx"] = best_idx; js["result"] = net1_result; js["username"] = username;
            string jss = js.dump();
            auto res = http->Post(url.c_str(), jss, "application/json");
            if (!res || res->status != 200) {
                cout << "error occured0" << endl; serrn++;
            }
            else {
                json hr = json::parse(res->body);
                if (hr["status"] == "error") cout << "error occured" << endl;
                else cout << "game is uploaded" << endl;
            }
        }
    }

	return make_tuple(net1_result, net1_result < 5 ? step : 0);
}

