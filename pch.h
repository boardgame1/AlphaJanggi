
#ifndef PCH_H
#define PCH_H

// 여기에 미리 컴파일하려는 헤더 추가

#define AllMoveLength 5225
#define MAX_TURN  202
#define BATCH_SIZE 100


using namespace std;
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <tuple>
#include <random>
#include <algorithm>
#include <numeric>
#include <thread>
#include <mutex>
#include <torch/torch.h>
#include <torch/script.h>
#include "httplib.h"
#include "Model.h"
#include "json.hpp"
using json = nlohmann::json;

#ifndef _WIN32
	#define sprintf_s sprintf
#endif

enum { KING = 1, CHA, PO, MA, SANG, SA, ZOL };

void actionTable(); string encode_lists(int pan[][9], int step);
void decode_binary(int pan[][9], string state_str); string toutf8(string codepage_str);
vector<int> possible_moves(string pan_str, int player, int step);
tuple<string, int> move(string pan_str, int move, int step);


extern int moveTable[], serrn; extern unordered_map<int, int> moveDict;
extern mt19937 rdgen; extern int pani[10][9]; extern uniform_real_distribution<float> urd;
extern Model model;

#endif //PCH_H
