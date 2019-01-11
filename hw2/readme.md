CKY Parser (CKY 분석기) 구현하는 hw2.

-실행 방법 및 주의 사항

동일한 디렉토리에 있는 input.txt(분석할 문장이 담긴 파일), grammar.txt(Chomsky Normal Form으로 작성된 context free grammar 파일)를 읽어 분석(parsing) 결과인 output.txt를 생성해 준다. 주의할 점은 프로그램에서 output.txt에 덮어쓰지않고 이어쓰기 때문에 여러 번 실행할 경우 output.txt에 여러 결과들이 모두 저장되어 알아보기 힘들 수 있다. 또 소스코드와 input.txt, grammar.txt는 동일 디렉토리에 있어야 한다.

-수행 과정
i) grammar.txt를 읽어서 grammar 규칙이 담긴 ‘gram_rule’ 이라는 dictionary파일을 생성한다.
이 때, CNF로 된 grammar를 split_line_gram에서 dictionary form 에 맞게끔 non-terminal symbol과 terminal symbol로 tokenize해서 저장시킨다. non-terminal symbol이 key값이 되고 terminal symbol들이 list를 이루어 item 값이 된다.
ii) 분석이 요구되는 문장을 list에 저장한다.
여백(‘ ‘)을 기준으로 문장을 tokenize하여 list에 저장한다.
iii) CKY Parsing algorithm을 적용시켜 문장을 parsing한다.
우선 table 2개를 만든다. 한 table은 조합을 통해 생성되는 결과인 non-terminal symbol을 저장하고 다른 table은 조합된 terminal symbol까지 모두 string의 형태로 저장해 놓는다. 이를 통해 앞선 table만을 이용, 조합의 결과(non-terminal symbol)를 얻는다. 그리고 얻은 결과를 바탕으로 terminal symbol까지 고려한 string을 생성, 저장한다.
iv) 최종 결과를 출력한다.
최종 결과는 table의 index 0, len(words) 에 저장되어 있다. len(words)는 입력 받은 문장의 길이를 의미한다. 이 때 ‘S’ symbol을 가진 경우만 출력을 하는데 이는 context-free grammar가 아닌 경우를 제외하기 위해서이다.
