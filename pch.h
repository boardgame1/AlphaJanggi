// pch.h: 미리 컴파일된 헤더 파일입니다.
// 아래 나열된 파일은 한 번만 컴파일되었으며, 향후 빌드에 대한 빌드 성능을 향상합니다.
// 코드 컴파일 및 여러 코드 검색 기능을 포함하여 IntelliSense 성능에도 영향을 미칩니다.
// 그러나 여기에 나열된 파일은 빌드 간 업데이트되는 경우 모두 다시 컴파일됩니다.
// 여기에 자주 업데이트할 파일을 추가하지 마세요. 그러면 성능이 저하됩니다.

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
void decode_binary(int pan[][9], string state_str);
vector<int> possible_moves(string pan_str, int player, int step);
tuple<string, int> move(string pan_str, int move, int step);


extern int moveTable[]; extern unordered_map<int, int> moveDict;
extern mt19937 rdgen; extern int pani[10][9]; extern uniform_real_distribution<float> urd;
extern Model model;

#endif //PCH_H
