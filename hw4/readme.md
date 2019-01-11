Skip-gram 을 이용하여 word embedding(word2vec) 을 만들고 여기에서 단어 사이의 유사도(similarity)를 구하는 hw4.



-구성

Step 1. Read corpus and tokenize words based on space 

Step 2. Build vocabulary using most common top k words, replace word with ids in the corpus

Step 3. 학습에 사용할 mini-batch 생성

Step 4. Build a skip-gram tensorflow graph

Step 5. Begin training

Step 5-1. query 파일을 읽어 target word를 저장한다.

Step 6. Restore checked tf checkpoint, perform query words similarity calculation



-코퍼스(말뭉치)

축약된 나무위키 코퍼스(국문)와 text8 코퍼스(영문)을 사용한다. 
