HMM 을 이용한 품사(POS, Part Of Speech) Tagger를 구현하는 hw3.

-명세
HMM(Hidden Markov Model)과 Viterbi 알고리즘을 이용하여 품사 태거를 제작하시오.
확률 값을 얻기 위한 데이터는 세종 형태소 말뭉치(train.txt)를 이용해야 하며, SMASH(형태소 분석기)를 사용한 결과인 후보 중에서 가장 높은 확률을 갖는 형태소/품사 열을 찾는다.

즉, 사실상 viterbi algorithm 을 구현하는 것.


-구성
프로그램은 크게 3단계로 구성된다. train.txt와 result.txt에서 데이터를 읽어드려 pre-processing까지 하는 ‘입력’ 부분과 앞서 얻은 데이터를 계산하여 확률값을 얻는 ‘Viterbi’부분 그리고 얻은 결과를 출력하는 ‘출력 부분이다.
프로그램은 HMM(Hidden Markov Model) 기반의 Viterbi 알고리즘을 이용하였고 SMASH(형태소 분석기)를 사용한 결과 중 가장 확률값이 높은 품사 태거를 선정, back-trace하여 태거들의 나열을 출력한다. 확률 값을 얻기 위한 데이터는 세종 형태소 말뭉치(train.txt)를 이용하였다.

입력부분
이 프로그램에서 확률은 Bigram기반 Maximum likelihood estimation으로 계산하기 때문에 word count가 중요한 부분을 차지한다. 그래서 앞서 언급했던 세종 형태소 말뭉치를 읽어들여 dictionary 자료구조에 저장한다. 항목은 아래와 같다.

pos = {} : 품사와 그 품사의 개수를 저장한다.
m_p = {} : 형태소/품사 묶음과 그 묶음의 개수를 저장한다.
p_p = {} : 품사/품사 묶음과 그 묶음의 개수를 저장한다.(품사/품사는 연속적으로 쓰인 두 품사)

구성을 위해서 우선 세종 형태소 말뭉치(train.txt)를 줄 별로 읽어 들였다. ‘\t’을 바탕으로 tokenize 한 후 우측부분을 취한다. 그리고 취한 부분을 다시 ‘+’로 tokenize하면 형태소/품사의 꼴을 얻을 수 있다. 주의할 점은 ‘+’ 이후에 형태소로 ‘+’가 올 경우인데 이 경우를 예외처리해 주었다. 품사를 얻기 위해서는 다시 ‘/’로 tokenize를 시켜주면 되었다. 역시 연속으로 ‘/’ 두개가 등장한 경우는 예외처리 하였다. 마지막으로 한 줄에 등장한 모든 품사들을 list에 넣은 후 이웃끼리 한 쌍이 되도록(예를 들어 4개의 품사일 경우 3개) 품사/품사로 묶어 주었다. 시작점(‘$’)를 체크하기 위해 문장의 시작 전에 한 칸 씩 뛰어져 있는 부분(“”)을 카운트하여 pos에 추가하였고 위에 형태소/품사를 묶은 방식과 유사하게 시작점(‘$’)과 그 다음 품사를 묶어서 p_p에 추가하였다.
Viterbi algorithm의 계산을 위해 SMASH로 뽑아낸 결과 result.txt를 줄 별로 읽어 들였다. result.txt는 어절별로 나뉘어 제공되기 때문에 읽어들인 어절의 생성확률을 계산하기 위한 사전 작업으로 위와 유사한 방법으로 tokenize한 후 저장하였다. 

Viterbi Algorithm 부분
우선 Viterbi algorithm을 적용하기 위해 어절의 생성확률을 구하였다. 어절의 생성확률은 명세서에서 제시한 아래의 그림을 토대로 계산하였다.
 
명세서 그림의 예를 통해 설명하자면 P(‘너|NP) * P(JKO|NP) * P(‘를’|JKO) 를 C( 너/NP )/C(NP) * C( NP,JKO )/C(NP) * C( 를/JKO ) / C(JKO)로 구하였다. 단, 조건에 따라 underflow가 0되는 것을 막기 위해 log scale로 변환해서 계산하였고 간단한 smoothing기법인 Laplace smoothing을 사용하였다. log scale은 math library의 log()함수를 사용하였고 Laplace smoothing은 아래와 같다.
 
결론적으로 어절의 생성 확률은 ln {C(너/NP)+1} – ln{C(NP)+V} + ln{C(NP,JKO)+1} – ln{C(NP)+V}  + ln{C(를/JKO)+1} – ln{C(JKO)+V}이다. 이때 V는 총 pos개수 이다.
Viterbi algorithm에서 천이 양상을 살펴 볼 때(확률 구할 때)는 transition probability를 곱해야하는데 이를 위해서 각 어절마다 첫 형태소의 품사와 마지막 형태소의 품사를 따로 저장하여 계산시 유용하게 하였다. 
사용한 리스트와 그에 대한 정보는 아래와 같다.
prob_of_phrases=[]: 이 리스트에서는 어절의 생성확률들을 저장
first_last_pos=[]   : 이 리스트에는 어절의 첫 형태소의 품사와 끝 형태소의 품사를 저장
index_history=[]   : Viterbi algorithm을 통해 max 확률을 갖도록 하는 이전 index를 저장
viterbi_cal=[]      : Viterbi algorithm으로 저장한 확률을 어절마다 저장
 
위의 리스트와 위의 Viterbi algorithm식을 이용하여 천이 과정의 확률들 중 max값을 취해 계산했다. 또 max로 선택된 index 역시 저장하였다.

출력 부분
Viterbi algorithm의 결과로 최종적으로 우측 끝 어절의 확률 중 max값을 취한 후 back-trace를 통해 가장 높은 확률을 갖는 천이 과정을 찾았다. 이러한 과정을 list에 저장하고 출력 시에는 저장한 방향과 거꾸로 출력하여 입력 받은 순대로 출력되도록 하였다. 결과 값은 output.txt에 저장하였다.
