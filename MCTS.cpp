#include "pch.h"
#include "MCTS.h"
#include "dirichlet.h"

MCTS::MCTS()
{
    c_puct = 1;
}

random_device rd;
mt19937 rdgen(rd());
uniform_real_distribution<float> urd(0, 1);

void MCTS::clear()
{
	visit_count.clear();
	svalue.clear();
	value_avg.clear();
	sprobs.clear();
}

tuple<float, string, int, vector<string>, vector<int>> MCTS::find_leaf(string state_int, int player, int step) {
    vector<string> states;
    vector<int> actions;
    string cur_state = move(state_int);
    int cur_player = player;
    float value = 9;

    while (value > 5 && !is_leaf(cur_state)) {
        states.push_back(cur_state);

        auto counts = visit_count[cur_state];
        float total_sqrt = accumulate(counts.begin(), counts.end(), 0);
        total_sqrt = sqrt(total_sqrt);
        auto probsl = sprobs[cur_state];
        auto values_avg = value_avg[cur_state];

        auto movel = possible_moves(cur_state, cur_player, step);
        const int alen = actions.size();
        vector<double> noises;
        if (alen < 1) {
            vector<double> dr; for (int i = 0; i < movel.size(); i++) dr.push_back(0.17);
            dirichlet_distribution<std::mt19937> d(dr); noises = d(rdgen);
        }
        float max_score = -INFINITY; int aidx, action;
        for (int i = 0; i < movel.size(); i++) {
            int m = movel[i];
            int idx = moveDict[m];
            float score = values_avg[idx] + c_puct * (alen > 0 ? probsl[idx] :
                0.75f * probsl[idx] + 0.25f * (float)noises[i]) * total_sqrt / (1 + counts[idx]);
            if (score > max_score) {
                max_score = score; aidx = idx; action = m;
            }
        }
        actions.push_back(aidx);
        int won;
        tie(cur_state, won) = move(cur_state, action, step);
        if (won > 0)
            // if somebody won the game, the value of the final state is -1 (as it is on opponent's turn)
            value = won - 1 == cur_player ? -1 : 1;
        cur_player = 1 - cur_player;
        step += 1;
    }

    if (value < 5) value *= 1 - (MAX_TURN - step) / 1000.f;
    return make_tuple( value, cur_state, step, states, actions );
}

bool MCTS::is_leaf(string& state_int) {
    return sprobs.find(state_int) == sprobs.end();
}

void MCTS::search_batch(int count, string& state_int, int player, torch::jit::script::Module* net, int step, torch::Device device) {
    for (int i = 0; i < count; i++)
        search_minibatch(state_int, player, net, step, device);
}

void MCTS::search_minibatch(string& state_int, int player, torch::jit::script::Module* net, int step, torch::Device device) {
    vector<tuple<float, vector<string>, vector<int>>> backup_queue;
    vector<string> expand_states;
    vector<int> expand_steps;
    vector<tuple<string, vector<string>, vector<int>>> expand_queue;
    unordered_set<string> planned;
    for (int i = 0; i < BATCH_SIZE; i++) {
        float value; string leaf_state; int leaf_step; vector<string> states; vector<int> actions;
        tie(value, leaf_state, leaf_step, states, actions) = find_leaf(state_int, player, step);
        if (value < 5)
            backup_queue.emplace_back(make_tuple(value, states, actions));
        else if (planned.find(leaf_state) == planned.end()) {
            planned.insert(leaf_state);
            expand_states.push_back(leaf_state);
            expand_steps.push_back(leaf_step);
            expand_queue.emplace_back(make_tuple(leaf_state, states, actions));
        }
    }
    if (!expand_queue.empty()) {
        auto batch_v = state_lists_to_batch(expand_states, expand_steps, device);
        vector<torch::jit::IValue> vt; vt.push_back(batch_v);
        auto t = net->forward(vt);
        at::Tensor logits_v = t.toTuple()->elements()[0].toTensor(), values_v = t.toTuple()->elements()[1].toTensor();
        at::Tensor probs_v = torch::nn::functional::softmax(logits_v, torch::nn::functional::SoftmaxFuncOptions(1));
        auto values = values_v.data().to(torch::kCPU);
        auto values_a = values.accessor<float, 2>();
        at::Tensor probs = probs_v.data().to(torch::kCPU);
        auto probs_a = probs.accessor<float, 2>();

    	for(int i=0;i<expand_queue.size();i++) {
            string leaf_state; vector<string> states; vector<int> actions;
            tie(leaf_state, states, actions) = expand_queue[i];
            float value = values_a[i][0];
            auto prob = probs_a[i];
            array<int, AllMoveLength> ii; ii.fill(0);
    		visit_count.insert(pair<string, array<int, AllMoveLength>>(leaf_state, ii));
            array<float, AllMoveLength> ff; ff.fill(0.f);
    		svalue.insert(pair<string, array<float, AllMoveLength>>(leaf_state, ff));
            value_avg.insert(pair<string, array<float, AllMoveLength>>(leaf_state, ff));
            array<float, AllMoveLength> p; for (int j = 0; j < AllMoveLength; j++) p[j] = prob[j];
            sprobs.insert(pair<string, array<float, AllMoveLength>>(leaf_state, p));
            backup_queue.emplace_back(make_tuple(value, states, actions));
        }
    }
    for (int i=0;i< backup_queue.size();i++) {
        float value; vector<string> states; vector<int> actions;
        tie(value, states, actions) = backup_queue[i];
        float cur_value = -value;
        for(int i=states.size()-1;i>=0;i--) {
            string state_int = states[i]; int action = actions[i];
            visit_count[state_int][action] += 1;
            svalue[state_int][action] += cur_value;
            value_avg[state_int][action] = svalue[state_int][action] / visit_count[state_int][action];
            cur_value = -cur_value;
		}
    }
}

tuple<array<float, AllMoveLength>, array<float, AllMoveLength>> MCTS::get_policy_value(const string &state_int, const vector<int>& movel, float tau) {
    array<int, AllMoveLength> counts; fill(counts.begin(), counts.end(), 0);
    for (int m : movel) {
        int idx = moveDict[m];
        counts[idx] = visit_count[state_int][idx];
    }
    int total = accumulate(counts.begin(), counts.end(), 0);
    array<float, AllMoveLength> probs; fill(probs.begin(), probs.end(), 0);
    if (tau == 0 || total < 1) {
        int n;
        if (total > 0) n = distance(counts.begin(), max_element(counts.begin(), counts.end()));
        else {
            uniform_int_distribution<int> dis(0, movel.size()-1);
            n = moveDict[movel[dis(rd)]];
        }
        probs[n] = 1.0f;
    }
    else {
        for (int i = 0; i < AllMoveLength; i++) {
            counts[i] = pow(counts[i], 1.0f / tau);
            probs[i] = counts[i] / (float)total;
        }
    }
    return make_tuple(probs, value_avg[state_int]);
}

