#pragma once
#include "pch.h"

class MCTS
{
public:
	unordered_map<string, array<int, AllMoveLength>> visit_count;
	unordered_map<string, array<float, AllMoveLength>> svalue;
	unordered_map<string, array<float, AllMoveLength>> value_avg;
	unordered_map<string, array<float, AllMoveLength>> sprobs;
	float c_puct;

	MCTS();
	void clear();
	tuple<float, string, int, vector<string>, vector<int>> find_leaf(string state_int, int player, int step);
	bool is_leaf(string& state_int);
	void search_batch(int count, string& state_int, int player, torch::jit::script::Module* net,
		int step, torch::Device device);
	void search_minibatch(string &state_int, int player, torch::jit::script::Module* net, int step,
		torch::Device device);
	tuple<array<float, AllMoveLength>, array<float, AllMoveLength>> get_policy_value(const string& state_int,
		const vector<int>& movel, float tau=1);
};

