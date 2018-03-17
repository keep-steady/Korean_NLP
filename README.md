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
unsupervised Learning. Similarity정의가 핵심<br/>

**비교사학습 복습**
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
    - 유사도 : n개의 데이터 X에 대하여 두 데이터 x_i, x_j간에 정의되는 임의의 거리 d(x_1, x_2)
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
    - Cluster ensemble에서 hierachical clustering을 활용하는 이유
        - 고차원에서는 유클리드/코사인을 이용하여 점과 점 사이의 거리(혹은 유사도)를 정의하기 어려움
        - Clustering Ensemble은 여러 번의 클러스터링 결과를 이용하여 점과 점 사이의 유사도를 잘 정의하였기 때문에 hierachical clustering이 가능
- DBSCAN(Density-Based Spatial Clustering of Application with Noise)
    - 모든 점이 반드시 그룹에 속하지 않는다고 가정(노이즈)
    - 유사도 : n개의 데이터 X에 대하여 두 데이터 x_i, x_j간에 정의되는 임의의 거리 d(x_i, x_j)
    - 그룹화 방식 : Threshold 이상의 밀도를 지닌 점들을 모두 이어나가는 방식
    - Parameters에 따라 결과가 민감하게 작동
        - 군집을 결정하는 밀도값 threshold에 의하여 데이터에서의 노이즈 비율이 예민하게 변경
    - 높은 계산 비용
- Graph(Community Detection)
    - K-Means, DBSCAN은 벡터로 표현된 데이터를 군집화하는 방법이지만, 그래프로 표현된 데이터에서의 군집화 방법
    
### 군집화
- K-means는 centroid를 중심으로 구형의 군집을 만듦
    - 유클리드의 구형은 공간에 구를 생성, 코싸인의 구형은 각도를 파티셔닝
- Hierachical clustering, DBSCAN은 임의의 모양의 군집을 추출하기 위한 방법
    - Sparse vector로 이뤄진 문서 공간은 복잡한 모양이 아님
    - 데이터가 복잡한 모양이 아니라는 가정을 할 수 있다면 단순한 알고리즘이 좋음
    
### 문서 군집화
- 문서를 BOW형식으로 표현할 경우 일반적인 군집화와 다른 특징을 지님
- 고차원 벡터에서는 매우 가까운 거리만 의미를 지님
    - 차원이 커질수록 거리가 1이라는 의미를 파악하기 어려움(거리가 1인 점들이 많아짐)
    - 고차원 벡터의 경우는 큰 k로 군집을 수행한 뒤, 동일한 의미를 지니는 군집들을 하나로 묶는 후처리(post-processing)이 필요
- Term vector에서 불필요한 단어들을 제거하는 것은 군집화 알고리즘에 도움됨


## 6. 벡터 인덱싱
### k-Nearest Neighbor<br/>
K-NN으로 다시 돌아옴. 가장 단순하고 직관적임. representation만 제대로 된다면 성능이 충분히 보장되는 알고리즘
 - Representation Learning : fc의 Softmax 등
 - 머신러닝의 핵심으로 돌아감. 좋은 representation?? 
    - Interpretablility : 벡터값으로부터 의미를 잘 이해할 수 있는지?, 시각화를 하였을 때 직관적 해석이 가능한가?
    - Discriminative power : 
        - (Classification) : 클래스별로 linear separable한가?
        - (Clustering, k-NN) : MECE한가? 비슷한 데이터는 유사도가 높고, 비슷하지 않은 데이터는 유사도가 낮으며, 이들의 차이는 명확히 구분되는지?
 - Nearest Neighbor Search -> Hashing으로 정리(Locally Sensitive Hashing:LSH)
 
### Locally Sensitive Hashing
- RP : https://en.wikipedia.org/wiki/Random_projection
- LSH(h_ij -> 유클리디언, 코싸인)
    - 고차원 벡터를 벡터간 거리를 보존하면서 저차원 벡터로 변환
    - RP에 의한 저차원 벡터를 integer로 바꾸면 각 벡터별로 integer vector형태의 label(hash code)를 만듬
    - Query가 들어오면, 동일한 RP mapper M을 곱하고, 그 결과를 integer vector로 만듦. 같은 integer vector를 지나는 벡터들만 실제 거리 계산
        - N개의 전체 데이터가 아닌 같은 hash code를 지니는 데이터만 거리를 계산
        - 같은 hash code를 지니는 데이터의 개수가 k보다 작을 경우, 비슷한 hash code를 지니는 데이터도 거리를 구할 수 있음
    - 한 개의 random projection에 의한 1차원 integer vector가 만드는 bucket은 한 종류의 평행한 선들에 의하여 구분되는 공간(기하학적 해석)
    - 두 개의 random projection에 의한 2차원 integer vector가 만드는 bucket은 두 종류의 평행한 선들에 의하여 구분되는 공간
    - Hash code의 길이가 길어질수록 bucket의 기하학적 모양은 구에 가까워지며, 구는 수학적으로 k-nearest neighbor search에 최적의 형태
    - r은 평행한 선분 간의 거리로, bucket안에 들어갈 점의 개수에 영향을 주며, 실제로 LSH를 이용할 때 가장 민감한 파라미터
    - 실제 LSH를 이용할 때는 하나의 g=(h_1, ..., h_m)가 아닌 여러 개의 g_j=(h_1,...,h_m)를 겹쳐서 사용
        - 하나의 g를 이용하면 nearest neighbor search의 성능이 매우 낮지만, 여러 장을 이용하면 어느 정도의 성능이 높음
        - 한 개의 g_1=(h_1,...,h_m)를 이용할 경우, bucket의 모서리에 query가 떨어지는 경우에 문제가 발생
        - 다른 g_2=(h_1,...,h_m)에 의해 생성된 bucket은 다른 데이터들을 포함할 수 있음. 두 개를 사용하면 조금더 정확한 search space
    
- Network based Indexer
    - LSH는 bucket안의 데이터 밀도가 다를 때 잘 대처를 하지 못하기 때문에 query point에 따라서 neighbor search정확도 및 검색 속도가 차이가 날 수 있음
    - Network based Indexer들은 query point에 관계없이 성능과 속도가 균일한 특성이 존재
    - small world phenomenon애 기반하기 때문에 균일한 속도가 나옴
- Inverted Indexer

 
