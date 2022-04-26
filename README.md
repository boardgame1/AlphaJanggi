# AlphaJanggi
An implementation of the AlphaZero algorithm for Janggi (Korean chess)

'알파 장기'는 AlphaZero 알고리즘을 적용한 장기 인공지능 프로그램입니다.<br>
사용 언어는 c++와 python이고 pytorch 라이브러리를 사용했습니다.<br>
<br>
자가 대국 생성하기<br>
```bash
Project1 -k self --cuda -t 3 -n 1
```
&nbsp;&nbsp; --cuda: nvidia 그래픽 카드 사용시<br>
&nbsp;&nbsp; -t: 쓰레드 갯수 (디폴트=1)<br>
&nbsp;&nbsp; -n: 그래픽 카드 번호 (디폴트=1)<br>
<br>
human_vs_ai는 텍스트 기반으로 인공지능과 대국할 수 있는 프로그램입니다.<br>
```bash
Project1 -k human --cuda -m best_1.pt
```
&nbsp;&nbsp; --cuda: nvidia 그래픽 카드 사용시<br>
&nbsp;&nbsp;  -m (모델 파일 이름): 사용하고자 하는 모델 파일(디폴트 best_model.pt)<br>
<br>

<h4>사용한 공개 소프트웨어</h4>
 json : https://github.com/nlohmann/json <br>
 http : https://github.com/yhirose/cpp-httplib <br>
 dirichlet : https://github.com/gcant/dirichlet-cpp <br>
