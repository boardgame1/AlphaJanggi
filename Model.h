#pragma once
#include "pch.h"
#include "MCTS.h"

void _encode_list_state(float dest_np[][10][9], int state_list[][9], int step);

torch::jit::IValue state_lists_to_batch(const vector<string>& state_lists, const vector<int>& steps_lists, torch::Device device);

tuple<int, int> play_game(int* value, shared_ptr<MCTS> mcts_stores, shared_ptr<MCTS> mcts_stores2, torch::jit::script::Module* net1,
    torch::jit::script::Module* net2, int steps_before_tau_0, int mcts_searches, int best_idx,
    string url, string cookie, torch::Device device, httplib::Client* http);