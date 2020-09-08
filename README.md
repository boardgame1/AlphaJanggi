# AlphaJanggi
An implementation of the AlphaZero algorithm for Janggi (Korean chess)

'알파 장기'는 AlphaZero 알고리즘을 적용한 장기 인공지능 프로그램입니다.<br>
사용 언어는 c++와 python이고 pytorch 라이브러리를 사용했습니다.<br>
웹 사이트: https://alphajanggi.net<br>
<br>
여러분의 참여로 이 프로그램의 장기 인공지능 실력을 높일 수 있습니다.<br>
self-play를 실행해 놓는 것만으로 알파 장기의 기력 향상에 도움을 줄 수 있습니다.<br>
```bash
Project1 -k self --cuda -t 3
```
&nbsp;&nbsp; -t: 쓰레드 갯수 (디폴트=1)<br>
&nbsp;&nbsp; --cuda: nvidia 그래픽 카드 사용시<br>
<br>
human_vs_ai는 텍스트 기반으로 인공지능과 대국할 수 있는 프로그램입니다.<br>
```bash
Project1 -k human --cuda -m best_1.pt
```
&nbsp;&nbsp; --cuda: nvidia 그래픽 카드 사용시<br>
&nbsp;&nbsp;  -m (모델 파일 이름): 사용하고자 하는 모델 파일(디폴트 best_model.pt)<br>
<br>
윈도우 실행 파일과 모델 파일은 위 웹 사이트에서 받을 수 있습니다.<br>

<h4>지원</h4>
 AI Hub http://aihub.or.kr : 컴퓨팅 자원<br>
 JetBrains https://www.jetbrains.com/ko-kr : 모든 jetbrains 제품의 1년 라이센스

<h4>사용한 공개 소프트웨어</h4>
 json : https://github.com/nlohmann/json <br>
 http : https://github.com/yhirose/cpp-httplib <br>
 dirichlet : https://github.com/gcant/dirichlet-cpp <br>
