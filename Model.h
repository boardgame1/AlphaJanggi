#pragma once
#include "pch.h"
#include "MCTS.h"

class Model
{
public:
	static void _encode_list_state(float dest_np[][10][9], int state_list[][9], int step);

	torch::jit::IValue state_lists_to_batch(vector<string> state_lists, vector<int> steps_lists, torch::Device device) const;

	tuple<int, int> play_game(int* value, shared_ptr<MCTS> mcts_stores, shared_ptr<MCTS> mcts_stores2, torch::jit::script::Module const net1,
		torch::jit::script::Module const net2, int steps_before_tau_0, int const mcts_searches, int best_idx,
		string url, string uname, torch::Device device, httplib::Client* http);
};

