// pch.cpp: 미리 컴파일된 헤더에 해당하는 소스 파일

#include "pch.h"

int moveTable[AllMoveLength];
unordered_map<int, int> moveDict;

void actionTable()
{
    int ms[][2] = { {-2, 1}, {-1, 2}, {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1},
        {-3, 2}, {-2, 3}, {2, 3}, {3, 2}, {3, -2}, {2, -3}, {-2, -3}, {-3, -2} };

    for (int m1 = 0; m1 < 58; m1++) {
        for (int n1 = 0; n1 < 10; n1++) {
			for (int l1 = 0; l1 < 9; l1++) {
				int src = n1 * 9 + l1, x, y;
				if (m1 < 9) { y = n1 - (m1 + 1); x = l1; }
				else if (m1 < 18) {
					y = n1 + (m1 - 8); x = l1;
				}
				else if (m1 < 26) { y = n1; x = l1 - (m1 - 17); }
				else if (m1 < 34) { y = n1; x = l1 + m1 - 25; }
				else if (m1 < 50) {
					y = n1 + ms[m1 - 34][0]; x = l1 + ms[m1 - 34][1];
				}
				else {
					if ((n1 < 3 || n1>6) && l1 > 2 && l1 < 6 && src % 2 == 1 - n1 / 5) {
						int k = (m1 - 50) % 2 + 1, a1 = m1 < 52 || m1>55 ? -1 : 1, a2 = m1 < 54 ? 1 : -1;
						y = n1 + k * a1; x = l1 + k * a2;
						if ((y > 2 && y < 7) || x < 3 || x>5) y = -1;
					}
					else y = -1;
				}
				int k = m1 * 90 + n1 * 9 + l1;
				moveTable[k] = y >= 0 && y < 10 && x >= 0 && x < 9 ? src * 100 + y * 9 + x : -1;
			}
        }
    }
    moveTable[5220]=0;
    moveTable[5221] = 10000;
    moveTable[5222] = 10001;
    moveTable[5223] = 10002;
    moveTable[5224] = 10003;
	for(int k=0;k<5225;k++)	if (moveTable[k] >= 0) moveDict.insert(pair<int, int>(moveTable[k], k));
}

string encode_lists(int pan[][9], int step) {
	string s; int b = 'a';
	for (int y = 0; y < 10;y++)
		for (int x = 0; x < 9;x++) s += char(pan[y][x] + b);
	return s + char(step);
}
void decode_binary(int pan[][9], string state_str) {
	int a = 0, b = 'a';
	for (int y = 0; y < 10;y++)
		for (int x = 0; x < 9; x++) {
			pan[y][x] = state_str.at(a) - b;
			a++;
		}
}

const int maap[][4] = { 0,-1,-1,-2, 0,-1,1,-2, 1,0,2,-1, 1,0,2,1, 0,1,1,2, 0,1,-1,2, -1,0,-2,-1, -1,0,-2,1 };
const int sangap[][6] = { 0,-1,-1,-2,-2,-3, 0,-1,1,-2,2,-3, 1,0,2,-1,3,-2, 1,0,2,1,3,2,
	0,1,1,2,2,3, 0,1,-1,2,-2,3, -1,0,-2,1,-3,2, -1,0,-2,-1,-3,-2 };
vector<int> possible_moves(string pan_str, int player, int step)
{
	vector<int>	moven;
	if (step < 2)
		for (int i = 0; i < 4; i++) moven.push_back(10000 + i);
	else {
		int pan[10][9]; decode_binary(pan, move(pan_str));
		for (int y = 0; y < 10; y++) {
			int  k = y / 7 * 7;
			for (int x = 0; x < 9; x++) {
				int ki = pan[y][x], i, j, alk = ki / 10, a;
				if (ki > 0 && alk == player)
					switch (ki % 10) {
					case KING: case SA:
						if (x == 4 && (y == 1 || y == 8))
							for (i = -1; i < 2; i++) for (j = 3; j < 6; j++) { a = pan[i + y][j]; if (a == 0 || a / 10 != alk) moven.push_back((y * 9 + x) * 100 + (i + y) * 9 + j); }
						else {
							for (i = 0; i < 3; i++) for (j = 3; j < 6; j++) if (abs(i + k - y) + abs(j - x) < 2 || (i == 1 && j == 4)) {
								a = pan[i + k][j]; if (a == 0 || a / 10 != alk)	moven.push_back((y * 9 + x) * 100 + (i + k) * 9 + j);
							}
						}
						 if (ki % 10 == KING) {
							 i = y; j = y > 5 ? -1 : 1; do i += j; while (i >= 0 && i < 10 && pan[i][x] == 0);
							 if (i >= 0 && i < 10 && pan[i][x] % 10 == KING) moven.push_back((y * 9 + x) * 100 + i * 9 + x);
						 }
						 break;
					case CHA:
						if (x == 4 && (y == 1 || y == 8)) {
							for (i = -1; i < 2; i += 2) for (j = 3; j < 6; j += 2) { a = pan[i + y][j]; if (a == 0 || a / 10 != alk) moven.push_back((y * 9 + x) * 100 + (i + y) * 9 + j); }
						}
						else if ((x == 3 || x == 5) && (y == k || y == k + 2)) {
							{a = pan[k + 1][4]; if (a == 0 || a / 10 != alk) moven.push_back((y * 9 + x) * 100 + (k + 1) * 9 + 4); }
							{int b = 2 * k + 2 - y; a = pan[b][8 - x]; if (pan[k + 1][4] == 0 && (a == 0 || a / 10 != alk))	moven.push_back((y * 9 + x) * 100 + b * 9 + 8 - x); }
						}
						for (i = x + 1; i < 9; i++) { a = pan[y][i]; if (a == 0 || a / 10 != alk) moven.push_back((y * 9 + x) * 100 + y * 9 + i); if (a > 0) break; }
						for (i = x - 1; i >= 0; i--) { a = pan[y][i]; if (a == 0 || a / 10 != alk) moven.push_back((y * 9 + x) * 100 + y * 9 + i); if (a > 0) break; }
						for (i = y + 1; i < 10; i++) { a = pan[i][x]; if (a == 0 || a / 10 != alk) moven.push_back((y * 9 + x) * 100 + i * 9 + x); if (a > 0) break; }
						for (i = y - 1; i >= 0; i--) { a = pan[i][x]; if (a == 0 || a / 10 != alk) moven.push_back((y * 9 + x) * 100 + i * 9 + x); if (a > 0) break; } break;
					case PO:
						for (i = x + 1; i < 8; i++) if (pan[y][i] > 0) break; if (i < 8 && pan[y][i] % 10 != PO) for (j = i + 1; j < 9; j++) {
							a = pan[y][j]; if (a == 0 || (a / 10 != alk && a % 10 != PO)) moven.push_back((y * 9 + x) * 100 + y * 9 + j); if (a > 0) break;
						}
						for (i = x - 1; i > 0; i--) if (pan[y][i] > 0) break; if (i > 0 && pan[y][i] % 10 != PO) for (j = i - 1; j >= 0; j--) {
							a = pan[y][j]; if (a == 0 || (a / 10 != alk && a % 10 != PO)) moven.push_back((y * 9 + x) * 100 + y * 9 + j); if (a > 0) break;
						}
						for (i = y + 1; i < 9; i++) if (pan[i][x] > 0) break; if (i < 9 && pan[i][x] % 10 != PO) for (j = i + 1; j < 10; j++) {
							a = pan[j][x]; if (a == 0 || (a / 10 != alk && a % 10 != PO)) moven.push_back((y * 9 + x) * 100 + j * 9 + x); if (a > 0) break;
						}
						for (i = y - 1; i > 0; i--) if (pan[i][x] > 0) break; if (i > 0 && pan[i][x] % 10 != PO) for (j = i - 1; j >= 0; j--) {
							a = pan[j][x]; if (a == 0 || (a / 10 != alk && a % 10 != PO)) moven.push_back((y * 9 + x) * 100 + j * 9 + x); if (a > 0) break;
						}
						if ((x == 3 || x == 5) && (y == k || y == k + 2)) {
							a = pan[k + 1][4]; int c = 2 * k + 2 - y, b = pan[c][8 - x];
							if (a > 0 && a % 10 != PO && (b == 0 || (b / 10 != alk && b % 10 != PO))) moven.push_back((y * 9 + x) * 100 + c * 9 + 8 - x);
						}
						break;
					case MA: for (i = 0; i < 8; i++) {
						int x1 = x + maap[i][2], y1 = y + maap[i][3]; if (y1 >= 0 && y1 < 10 && x1 >= 0 && x1 < 9) {
							a = pan[y1][x1]; if ((a == 0 || a / 10 != alk) && pan[y + maap[i][1]][x + maap[i][0]] == 0) moven.push_back((y * 9 + x) * 100 + y1 * 9 + x1);
						}
					} break;
					case SANG: for (i = 0; i < 8; i++) {
						int x1 = x + sangap[i][4], y1 = y + sangap[i][5];	if (y1 >= 0 && y1 < 10 && x1 >= 0 && x1 < 9) {
							a = pan[y1][x1]; if ((a == 0 || a / 10 != alk) && pan[y + sangap[i][1]][x + sangap[i][0]] == 0 && pan[y + sangap[i][3]][x + sangap[i][2]] == 0)
								moven.push_back((y * 9 + x) * 100 + y1 * 9 + x1);
						}
					} break;
					case ZOL:
						int ad = alk == 1 ? -1 : 1;
						for (i = -1; i < 2; i++) {
							int a, x1 = x + i, y1 = y + (i == 0 ? ad : 0);
							if (x1 >= 0 && x1 < 9 && y1 >= 0 && y1 < 10) { a = pan[y1][x1]; if (a == 0 || a / 10 != alk) moven.push_back((y * 9 + x) * 100 + y1 * 9 + x1); }
						}
						if (x == 4 && y == k + 1) for (i = -1; i < 2; i += 2) {
							int b = y + ad; a = pan[b][x + i]; if (a == 0 || a / 10 != alk)	moven.push_back((y * 9 + x) * 100 + b * 9 + x + i);
						}
						if ((x == 3 || x == 5) && (y == 2 || y == 7)) { a = pan[k + 1][4]; if (a == 0 || a / 10 != alk) moven.push_back((y * 9 + x) * 100 + (k + 1) * 9 + 4); }
					}
			}
		}
		moven.push_back(0);
	}
	return moven;
}

const int pieceScore[] = { 13, 7, 5, 3, 3, 2 };

int _endWin(int pan[][9]) {
	float pscore[] = { 0, 1.5 };
	for (int y = 0; y < 10;y++) 
		for (int x = 0; x < 9; x++) {
			int ki = pan[y][x];
			if (ki % 10 > 1) pscore[ki / 10] += pieceScore[ki % 10 - 2];
		}
	return pscore[0] > pscore[1] ? 1 : 2;
}

tuple<string, int> move(string pan_str, int move, int step) {
	int pan[10][9]; decode_binary(pan, std::move(pan_str));
	if (move >= 10000) {
		if (step < 1)
			if (move == 10000) {
				pan[9][1] = 14; pan[9][2] = 15; pan[9][6] = 14; pan[9][7] = 15;
			}
			else if (move == 10001) {
				pan[9][1] = 15; pan[9][2] = 14; pan[9][6] = 15; pan[9][7] = 14;
			}
			else if (move == 10002) {
				pan[9][1] = 14; pan[9][2] = 15; pan[9][6] = 15; pan[9][7] = 14;
			}
			else {
				pan[9][1] = 15; pan[9][2] = 14; pan[9][6] = 14; pan[9][7] = 15;
			}
		else
			if (move == 10000) {
				pan[0][1] = 5; pan[0][2] = 4; pan[0][6] = 5; pan[0][7] = 4;
			}
			else if (move == 10001) {
				pan[0][1] = 4; pan[0][2] = 5; pan[0][6] = 4; pan[0][7] = 5;
			}
			else if (move == 10002) {
				pan[0][1] = 4; pan[0][2] = 5; pan[0][6] = 5; pan[0][7] = 4;
			}
			else {
				pan[0][1] = 5; pan[0][2] = 4; pan[0][6] = 4; pan[0][7] = 5;
			}
		return make_tuple( encode_lists(pan, step + 1), 0 );
	}
	if (move == 0)
		return make_tuple( encode_lists(pan, step + 1), step < MAX_TURN - 1? 0: _endWin(pan) );
	
	int spos = move / 100, tpos = move % 100, y0 = spos / 9, x0 = spos % 9, y1 = tpos / 9, x1 = tpos % 9;
	int captured = pan[y1][x1];
	int piece = pan[y0][x0];
	pan[y1][x1] = piece;
	pan[y0][x0] = 0;

	return make_tuple( encode_lists(pan, step + 1), captured % 10 == KING && piece % 10 != KING ? 2 - captured / 10 :
		step < MAX_TURN - 1 && captured % 10 != KING ? 0 : _endWin(pan) );
}

string toutf8(string codepage_str) {
#ifdef _WIN32
	int size = MultiByteToWideChar(CP_ACP, MB_COMPOSITE, codepage_str.c_str(),
		codepage_str.length(), nullptr, 0);
	std::wstring utf16_str(size, '\0');
	MultiByteToWideChar(CP_ACP, MB_COMPOSITE, codepage_str.c_str(),
		codepage_str.length(), &utf16_str[0], size);

	int utf8_size = WideCharToMultiByte(CP_UTF8, 0, utf16_str.c_str(),
		utf16_str.length(), nullptr, 0,
		nullptr, nullptr);
	std::string utf8_str(utf8_size, '\0');
	WideCharToMultiByte(CP_UTF8, 0, utf16_str.c_str(),
		utf16_str.length(), &utf8_str[0], utf8_size,
		nullptr, nullptr);
	return utf8_str;
#else
	return codepage_str;
#endif
}
