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
    - Spacing correct(count base)
    - L-Tokenizer / MaxScoreTokenizer / KoNLPy

### 명사추출
Sequence Labeling
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


## 4. 단어/문서 임베딩
Embedding : 어떤 정보를 보존하며 저차원의 dense vector로 학습하는가
 - Word2Vec : 단어 주변에 등장하는 다른 단어들의 분포의 유사성을 보존하는 벡터 표현법
    - CBOW : 주위 단어로 현재 단어 w(t)를 예측하는 모델
    - Skipgram : 현재 단어 w(t)로 주위 단어 모두를 예측하는 모델
    - |v| class Softmax regression을 사용하여 각 단어의 벡터를 학습시키는 classifier
 - Doc2Vec
 - Glove
 - Fasttext
 
 
## 5. 문서 군집화
unsupervised Learning. Similarity정의가 핵심
 - (spherical) K-Means
    - K-Means : 
    - spherikal K-Means : Euclidean이 아닌 Cosine distance
 - GMM(Gaussian Mixture Model)
    - 유사도 : n개의 데이터 X 대하여 k개의 Gaussian distribution의 확률값 사용
        - 데이터의 분포(밀도)를 고려한 K-means, 유클리드 거리에 대해서만 정의됨
    - 그룹화 방식 : 데이터의 분포를 잘 설명할 수 있는 k개의 Gaussian distribution parameter를 학습
    - 데이터가 Centroid를 중심으로 Gaussian distribution을 띄는 분포를 따른다는 가정
        - 밀도 차이가 있는 데이터 뭉치들이 있는 경우에 적합
    - 원형이 아닌 데이터 분포의 모양도 가능함
 - BGMM(Bayesian Gaussian Mixture Model)
    - 유사도 : n개의 데이터 X에 대하여 모델이 학습하는 가장 적절한 K개의 Gaussian distribution의 확률값
    - 그룹화 방식 : 
        - 데이터의 분포를 잘 설명할 수 있는 k개의 Gaussian distribution parameter(mu, sigma)를 학습
        - Dirichlet process를 이용하여 가장 적절한 군집의 갯수(Gaussian Model의 갯수)를 함께 학습함. 군집갯수X
- Hierarachical clustering
    - 유사도 : n개의 데이터 X에 대하여 두 데이터 x_1, x_2간에 정의되는 임의의 거리 d(x_1, x_2)
        - 그룹 간의 거리는 d(C_i, C_j)를 기반으로 정의(min, max, average 등..)
        - 하나의 그룹 C_i는 1개 이상의 데이터로 이뤄짐
        - single linkage
        - complete linkage
    - 그룹화 방식 : 그룹의 수는 정하지 않음. 거리가 가장 가까운 점들을 하나의 집합으로 묶어감
    - Outliers를 알아서 걸를 수 있음
        - Single linkage는 가장 가까운 점들을 하나씩 이어나가는 구조이기 때문에, 다른 점들이 큰 군집으로 묶여갈 때까지 다른 점들과 잘 묶이지 않는 점으로 Outlier로 판단할 수 있음
    - 고차원 벡터에서 잘 작동하지 않음
        - 고차원에서는 최근접이웃들의 거리 외에는 정보력이 없음
        - Complete linkage를 이용할 경우, 군집 안에 포함된 모든 점들간의 거리의 평균을 두 군집 간의 거리로 이용. 대부분 점들의 거리가 멀 경우, 군집간 거리가 잘 정의되지 않음
- DBSCAN
