
#include "pch.h"
#include <iostream>
#include "MCTS.h"
#ifdef _WIN32
	#include "getopt.h"
	#include <conio.h>
#else
	#include <getopt.h>
#endif
const char* domain = "alphajanggi.net"; string SURL = "/selfplay12";

string piece_str = "초차포마상사졸漢車包馬象士兵";
void render(string pan_str, int player_human) {
	int pan[10][9];  decode_binary(pan, pan_str);
	cout << "   1  2  3  4  5  6  7  8  9" << endl;
	for (int y = 0; y < 10; y++) {
		string s = char((y > 0 ? 10 - y : 0) + '0') + string(" ");
		for (int x = 0; x < 9; x++) {
			int a = pan[player_human > 0 ? y : 9 - y][player_human > 0 ? x : 8 - x];
			s += a > 0 ? piece_str.substr((a / 10 * 7 + int(a % 10) - 1) * 2, 2) + (x < 8 ? "-" : " ") :
				string(y < 1 ? (x < 1 ? "┌" : x < 8 ? "┬" : "┐") :
					y < 9 ? (x < 1 ? "├" : x < 8 ? "╋" : "┤") :
					(x < 1 ? "└" : x < 8 ? "┴" : "┘")) + (x < 8 ? "─-" : " ");
		}
		cout << s << endl;
		if (y < 9)
			cout << (y < 1 or y == 7 ? "  │  │  │  │＼│／│  │  │  │ " : y == 1 or y>7 ? "  │  │  │  │／│＼│  │  │  │ "
				: "  │  │  │  │  │  │  │  │  │ ") << endl;
	}
}

string masang[] = { "마상마상", "상마상마", "마상상마", "상마마상" };
int mcts_searches = 60;
const int LEVELC = 160;
int pani[10][9] = { {2, 0, 0, 6, 0, 6, 0, 0, 2}, {0, 0, 0, 0, 1, 0, 0, 0, 0}, {0, 3, 0, 0, 0, 0, 0, 3, 0}, {7, 0, 7, 0, 7, 0, 7, 0, 7},
	{0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0, 0}, {17, 0, 17, 0, 17, 0, 17, 0, 17}, {0, 13, 0, 0, 0, 0, 0, 13, 0},
	{0, 0, 0, 0, 11, 0, 0, 0, 0}, {12, 0, 0, 16, 0, 16, 0, 0, 12} };

void play_game(torch::jit::script::Module& net1, int steps_before_tau_0, torch::Device device) {
	string pan = encode_lists(pani, 0);
	vector<string> historystr;
	int cur_player = 0;
	int step = 0;
	MCTS mctsi = MCTS();

	bool exitf = false;
	int a0 = '1', player_human;
	while (1) {
		cout << "플레이하려는 진영을 선택하세요 0) 초, 1)한 ?";
		string s; getline(cin, s);
		if (s.find("level") != string::npos) {
			mcts_searches = LEVELC * stoi(s.substr(6));
			cout << "OK" << endl;
		}
		else {
			player_human = stoi(s) < 1 ? 0 : 1;
			break;
		}
	}

	while (1) {
		vector<int> movelist = possible_moves(pan, cur_player, step);
		if (step > 9 && historystr.end()[-4].substr(90) == historystr.end()[-8].substr(90)) {
			int p[10][9]; decode_binary(p, pan);
			for (int idx = 0; idx < movelist.size(); idx++) {
				int m = movelist[idx];
				int spos = m / 100, tpos = m % 100, y0 = spos / 9, x0 = spos % 9, y1 = tpos / 9, x1 = tpos % 9;
				int captured = p[y1][x1]; p[y1][x1] = p[y0][x0]; p[y0][x0] = 0;
				string ps = encode_lists(p, step + 1);
				if (ps.substr(90) == historystr.end()[-4].substr(90)) {
					movelist.erase(movelist.begin() + idx); break;
				}
				p[y0][x0] = p[y1][x1]; p[y1][x1] = captured;
			}
		}
		int action;
		if ((step < 2 && cur_player != player_human) || (step > 1 && cur_player == player_human)) {
			if (step < 2)
				cout << "마상 차림을 선택하세요 0) " + masang[0] + ", 1) " + masang[1] + ", 2) " + masang[2] + ", 3) " + masang[3] << endl;
			else {
				render(pan, player_human);
				if (step == 2 || step == 3)
					cout << endl << "옮기고자 하는 기물의 세로 번호, 가로 번호, 목적지의 세로 번호, 가로 번호 ex) 0010  한수 쉬기: 0" << endl;
			}
			action = -1;
			while (action < 0) {
				cout << (step > 1 ? to_string(step - 1) : "") + string(" ? ");
				string s; getline(cin, s);
				if (s == "new") {
					exitf = true; break;
				}
				if (s.find("level") != string::npos) {
					mcts_searches = LEVELC * stoi(s.substr(6));
					cout << "OK" << endl;
				}
				else if (step < 2) {
					if (s.length() == 1 && s.at(0) >= '0' && s.at(0) < '4') action = stoi(s) + 10000;
				}
				else if (s.length() == 1) action = 0;
				else if (s == "undo" && step > 3) {
					step -= 2; historystr.pop_back(); historystr.pop_back(); pan = historystr.back();
					movelist = possible_moves(pan, cur_player, step);
					render(pan, player_human);
				}
				else if (s.length() == 4 && s[0] >= '0' && s[0] <= '9' && s[1] > '0' && s[1] <= '9' && s[2] >= '0' && s[2] <= '9' && s[3] > '0' && s[3] <= '9') {
					int b1 = s[0] > '0' ? 9 - s[0] + a0 : 0;
					if (player_human < 1) b1 = 9 - b1;
					int b2 = s[1] - a0;
					if (player_human < 1) b2 = 8 - b2;
					int b3 = s[2] > '0' ? 9 - s[2] + a0 : 0;
					if (player_human < 1) b3 = 9 - b3;
					int b4 = s[3] - a0;
					if (player_human < 1) b4 = 8 - b4;
					action = (b1 * 9 + b2) * 100 + b3 * 9 + b4;
				}
				if (find(movelist.begin(), movelist.end(), action) == movelist.end()) action = -1;
				else cout << "OK" << endl;
			}
		}
		else {
			mctsi.clear();
			mctsi.search_batch(mcts_searches, pan, cur_player, net1, step, device);
			array<float, AllMoveLength> probs, values;
			tie(probs, values) = mctsi.get_policy_value(pan, movelist);
			int n;
			if (step < steps_before_tau_0) {
				float tt = 0, f = urd(rdgen);
				for (int n0 : movelist) {
					n = moveDict[n0];
					tt += probs[n];
					if (tt >= f) break;
				}
			}
			else n = distance(probs.begin(), max_element(probs.begin(), probs.end()));
			action = moveTable[n];
#ifdef _DEBUG
			for (int m : movelist) {
				char buf[50]; sprintf_s(buf, "%04d %.2f,  ", m, probs[moveDict[m]]); cout << buf;
			}
			cout << endl;
#endif
			if (step < 2) {
				cout << (step < 1 ? "한: " : "초: ") + masang[action - 10000] + ' ' + to_string(values[n]) << endl;
				if (step == 1) render(pan, player_human);
			}
			else {
				if (action < 1) cout << "한수쉼 " + to_string(values[n]) << endl;
				else {
					int b1 = action / 100 / 9;
					if (player_human < 1) b1 = 9 - b1;
					int b2 = action / 100 % 9;
					if (player_human < 1) b2 = 8 - b2;
					int b3 = action % 100 / 9;
					if (player_human < 1) b3 = 9 - b3;
					int b4 = action % 100 % 9;
					if (player_human < 1) b4 = 8 - b4;
					char s[6];
					s[0] = b1 > 0 ? 9 - b1 + a0 : '0'; s[1] = b2 + a0; s[2] = b3 > 0 ? 9 - b3 + a0 : '0'; s[3] = b4 + a0; s[4] = 0;
					cout << s + string(" ") + to_string(values[n]) << endl;
				}
			}
		}
		if (exitf) break;
		int won;
		tie(pan, won) = move(pan, action, step);
		historystr.push_back(pan);
		if (won > 0) {
			render(pan, player_human);
			cout << (won == 1 ? "초" : "한") + string(" 승") << endl;
			break;
		}
		cur_player = 1 - cur_player;
		step += 1;
	}
}

void play(int* val, mutex& mtx, torch::jit::script::Module net, int best_idx, string username, torch::Device device,
	int step_idx, int *done, httplib::Client* http) {
	shared_ptr<MCTS> mcts_store = make_shared<MCTS>();
	while (1) {
		chrono::steady_clock::time_point begin = chrono::steady_clock::now();
		int a, game_steps;
		tie(a, game_steps) = model.play_game(val, mcts_store, nullptr, net, net, 20,
			20, best_idx, SURL, username, device, http);
		chrono::steady_clock::time_point end = chrono::steady_clock::now();
		float dt = chrono::duration_cast<chrono::milliseconds>(end - begin).count() / 1000.f;
		float speed_steps = game_steps / dt;

		bool bf = false;
		lock_guard<std::mutex> lock(mtx);
		if (game_steps > 0)  val[1] += 1;
		if (val[0] <= 0) bf = true;
		if (game_steps > 0) {
			char s[90];
			sprintf_s(s, "Step %d, steps %3d, steps/s %5.2f, best_idx %d\n", step_idx + val[1], game_steps, speed_steps, best_idx);
			cout << s;
		}
		if (bf) {
			*done = 1; break;
		}
	}
}

#ifndef _WIN32
#include <unistd.h>
#include <termios.h>

int getch()
{
	int c;
	struct termios oldattr, newattr;

	tcgetattr(STDIN_FILENO, &oldattr);           // 현재 터미널 설정 읽음
	newattr = oldattr;
	newattr.c_lflag &= ~(ICANON | ECHO);         // CANONICAL과 ECHO 끔
	newattr.c_cc[VMIN] = 1;                      // 최소 입력 문자 수를 1로 설정
	newattr.c_cc[VTIME] = 0;                     // 최소 읽기 대기 시간을 0으로 설정
	tcsetattr(STDIN_FILENO, TCSANOW, &newattr);  // 터미널에 설정 입력
	c = getchar();                               // 키보드 입력 읽음
	tcsetattr(STDIN_FILENO, TCSANOW, &oldattr);  // 원래의 설정으로 복구
	return c;
}
#endif

string getpasswd() {
	char password1[100];
	for (int I = 0; I < 100; I++) {
		password1[I] = getch();
		if (password1[I]==10 || password1[I] == 13) {
			password1[I] = 0; cout << endl; break;
		}
		cout << '*';
	}
	return password1;
}

int main(int argc, char** argv)
{
	int opt;
	struct option longopts[] = {
		{ "cuda", 0, nullptr, 'c' },
		{ "model", 1, nullptr, 'm' },
		{ "model2", 1, nullptr, 'n' },
		{ "kind", 1, nullptr, 'k' },
		{ "thread", 1, nullptr, 't' },
		{ 0, 0, 0, 0 }
	};
	bool cudaf = false; int num_thread = 1;
	string modelfile = "./best_model.pt", modelfile2, kind = "self";
	while ((opt = getopt_long(argc, argv, "cm:k:n:t:", longopts, nullptr)) != -1)
	{
		switch (opt)
		{
		case 'c': cudaf = true; break;
		case 'm': modelfile = optarg; break;
		case 'n': modelfile2 = optarg; break;
		case 'k': kind = optarg; break;;
		case 't': num_thread = stoi(optarg); break;;
		default: break;
		}
	}

	const torch::Device device = cudaf ? torch::kCUDA : torch::kCPU;
	cout << device << endl;

	actionTable();
	torch::jit::script::Module net;
	if (kind != "self") {
		ifstream f(modelfile.c_str());
		if (f.good()) {
			net = torch::jit::load(modelfile);
			net.to(device);
			net.train(false);
		}
		else {
			cout << modelfile + " 파일이 존재하지 않습니다"; return 1;
		}
	}
	
	if (kind == "human")
		while (1)
			play_game(net, 7, device);
	if (kind == "ai") {
		ifstream f(modelfile2.c_str());
		torch::jit::script::Module net2;
		if (f.good()) {
			net2 = torch::jit::load(modelfile2);
			net2.to(device);
			net2.train(false);
		}
		else {
			cout << modelfile2 + " 파일이 존재하지 않습니다"; return 1;
		}
		for (int k = 0; k < 2; k++) {
			int wins = 0, losses = 0, draws = 0;
			for (int i = 0; i < 10; i++) {
				int r, a;
				tie(r, a) = model.play_game(nullptr, nullptr, nullptr, k < 1 ? net : net2, k < 1 ? net2 : net,
					MAX_TURN, 50, -1, "", "", device, nullptr);
				cout << r << endl;
				if (r > 0)
					wins++;
				else if (r < 0)
					losses += 1;
				else
					draws += 1;
			}
			string name_1 = k < 1 ? modelfile : modelfile2, name_2 = k < 1 ? modelfile2 : modelfile;
			char buf[512]; sprintf(buf, "%s vs %s -> w=%d, l=%d, d=%d\n", name_1.c_str(), name_2.c_str(), wins, losses, draws);
			cout << buf;
		}
	}else if(kind == "self") {
		httplib::Client *http = new httplib::Client(domain);
		string username, password;
		while (1) {
			bool createf = false;
			cout << "user ID (to create, enter 0): "; getline(cin, username);
			if (username.empty()) continue;
			if (username == "0") {
				cout << "user ID to create: "; getline(cin, username);
				if (username.empty()) continue;
				cout << "password: ";
				string password1 = getpasswd();
				cout << "password to confirm: ";
				string password2 = getpasswd();
				if (password1.empty() || password1!=password2) continue;
				createf = true; password = password1;
			}
			else {
				cout << "password: ";
				string password1 = getpasswd();
				if (password1.empty()) continue;
				password = password1;
			}
			string js = R"({ "username":")" + username + R"(", "password" :")" + password + R"(", "createf" :)" + string(createf ? "true" : "false") + " }";
			auto result = http->Post("/user12", js, "application/json");
			if(!result || result->status != 200) {
				cout << "문제가 지속되면 프로젝트 사이트에서 프로그램을 다시 다운로드하세요.";
				return 0;
			}
			json hr = json::parse(result->body);
			if (hr["status"] == "ok") break;
			if (hr["status"] == "dup") {
				cout << "duplicate user ID" << endl; continue;
			}
			if (hr["status"] == "notexist")
				cout << "user ID does not exist or password is incorrect" << endl;
		}
		int step_idx = 0;
		mutex mtx;
			
		while (1) {
			cout << "checking model" << endl;
			int best_idx = 0;
			int mhash = -1;
			string dfile = "./selfmodel.dat";
			ifstream f(modelfile.c_str()), df(dfile.c_str());
			if (f.good() && df.good()) {
				string s; getline(df, s); df.close();
				mhash = stoi(s.substr(0, 8), nullptr, 16);
				best_idx = stoi(s.substr(8), nullptr, 16);
			}
			string ws = "/checkmodel";
			if (mhash >= 0) ws += "?mhash=" + to_string(mhash);
			auto res = http->Get(ws.c_str());
			if (!res || res->status != 200) return 1;
			json hr = json::parse(res->body);
			if (hr["status"] == "download") {
				string ur = hr["url"];
				ofstream myfile; int total = 0;
				myfile.open(modelfile, ios::out | ios::trunc | ios::binary);
				if (!myfile.is_open()) cout << "Unable to open file";
				auto res = http->Get(ur.c_str(),
					[&](const char* data, size_t data_length) {
					myfile.write(data, data_length);
					bool b = total / 1'000'000 != (total + data_length) / 1'000'000;
					total += data_length;
					if (b) cout << "\r " << total / 1'000'000 << " M" << flush;
					return true;
				});
				cout << endl; myfile.close();
				ofstream df(dfile, ios::out | ios::trunc);
				char s[50]; int hash = hr["hash"], idx = hr["idx"]; sprintf_s(s, "%08x%x", hash, idx);
				best_idx = idx;
				df << s;
				df.close();
			}
			
			net = torch::jit::load(modelfile);
			net.to(device);
			net.train(false);

			vector<thread>	processes; int* mar = new int[2]; int* done=new int[num_thread]; mar[0] = 1; mar[1] = 0;
			for (int i = 0; i < num_thread; i++) {
				done[i] = 0;
				processes.emplace_back(thread(play, mar, ref(mtx), net, best_idx, username, device,
					step_idx, &done[i], http));
			}
			while (1) {
				chrono::milliseconds timespan(500); this_thread::sleep_for(timespan);
				lock_guard<std::mutex> lock(mtx);
				if (mar[0] > 0 && mar[1] >= 30 * num_thread) mar[0] = 0;
				int i;
				for (i = 0; i < num_thread;i++) if(done[i]<1) break;
				if (i == num_thread)
					break;
			}
			for (int i = 0; i < processes.size();i++) processes[i].join();
			step_idx += mar[1]; delete[] mar, done;
			cout << endl;
			delete http; http = new httplib::Client(domain);
		}
	}
}
