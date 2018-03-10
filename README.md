# Korean_NLP(한국어 자연어처리)
---
참조자료 : https://github.com/lovit/soynlp

## 1. 한국어 단어추출 및 명사추출 Tokenizing
### 단어추출(통계 기반 단어 추출)
- Cohesion Probability(interior) : 부분 어절 판별 함수
    - L-Tokenizer(어절의 왼쪽 부분), MaxScoreTokenizer(띄어쓰기 오류를 포함하는 경우)
    - RegexTokenizer
- Branching Probability(Accessor Variety)
- Tokenize단계
    - Nomalizer
    - Regex tokenizer
    - L-Tokenizer / MaxScoreTokenizer / KoNLPy

### 명사추출
- HMM(Hidden Markov Model)
- MEMM(Maximum Entropy Markov Model)
- Conditional Random Field(CRF) 


## 2. 한국어 띄어쓰기/오탈자 교정
### 오탈자 제거(String distance)
- Levenshteion(Edit) distance
- Jaccard
- Cosine

## 3. 문서 분류
- Logistic Regression(Softmax Regression)
- Reural Networks
- Regularization(Lasso, Ridge)
    - L2 regularization은 lambda가 작아짐에 따라(=beta/max(beta)가 커짐에 따라) 많은 beta들이 동시에 커짐
    - L1 regularization은 lambda가 작아짐에 따라 변수를 추가적으로 이용함
