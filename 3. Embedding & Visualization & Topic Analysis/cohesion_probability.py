from collections import defaultdict
from collections import namedtuple
import numpy as np

class CohesionProbability:
    
    def __init__(self, left_min_length=1, left_max_length=10, right_min_length=1, right_max_length=6):

        self.left_min_length = left_min_length
        self.left_max_length = left_max_length
        self.right_min_length = right_min_length
        self.right_max_length = right_max_length

        self.L = defaultdict(int)
        self.R = defaultdict(int)

    def get_cohesion_probability(self, word):

        if not word:
            return (0, 0, 0, 0)

        word_len = len(word)

        l_freq = 0 if not word in self.L else self.L[word]
        r_freq = 0 if not word in self.R else self.R[word]

        if word_len == 1:
            return (0, 0, l_freq, r_freq)

        l_cohesion = 0
        r_cohesion = 0

        # forward cohesion probability (L)
        if (self.left_min_length <= word_len) and (word_len <= self.left_max_length):

            l_sub = word[:self.left_min_length]
            l_sub_freq = 0 if not l_sub in self.L else self.L[l_sub]

            if l_sub_freq > 0:
                l_cohesion = np.power((l_freq / float(l_sub_freq)), (1 / (word_len - len(l_sub) + 1.0)))

        # backward cohesion probability (R)
        if (self.right_min_length <= word_len) and (word_len <= self.right_max_length):

            r_sub = word[-1 * self.right_min_length:]
            r_sub_freq = 0 if not r_sub in self.R else self.R[r_sub]

            if r_sub_freq > 0:
                r_cohesion = np.power((r_freq / float(r_sub_freq)), (1 / (word_len - len(r_sub) + 1.0)))

        return (l_cohesion, r_cohesion, l_freq, r_freq)

    def get_all_cohesion_probabilities(self):

        cp = {}
        words = set(self.L.keys())
        for word in self.R.keys():
            words.add(word)

        for word in words:
            cp[word] = self.get_cohesion_probability(word)

        return cp

    def counter_size(self):
        return (len(self.L), len(self.R))

    def prune_extreme_case(self, min_count):

        before_size = self.counter_size()
        self.L = defaultdict(int, {k: v for k, v in self.L.items() if v > min_count})
        self.R = defaultdict(int, {k: v for k, v in self.R.items() if v > min_count})
        after_size = self.counter_size()

        return (before_size, after_size)

    def train(self, sents, num_for_pruning=0, min_count=5):

        for num_sent, sent in enumerate(sents):
            for word in sent.split():

                if not word:
                    continue

                word_len = len(word)

                for i in range(self.left_min_length, min(self.left_max_length, word_len) + 1):
                    self.L[word[:i]] += 1

                # for i in range(self.right_min_length, min(self.right_max_length, word_len)+1):
                for i in range(self.right_min_length, min(self.right_max_length, word_len)):
                    self.R[word[-i:]] += 1

            if (num_for_pruning > 0) and ((num_sent + 1) % num_for_pruning == 0):
                self.prune_extreme_case(min_count)

        if (num_for_pruning > 0) and ((num_sent + 1) % num_for_pruning == 0):
            self.prune_extreme_case(min_count)

    def extract(self, min_count=5, min_cohesion=(0.05, 0), min_droprate=0.8, remove_subword=True):

        word_to_score = self.get_all_cohesion_probabilities()
        word_to_score = {word: score for word, score in word_to_score.items()
                         if (score[0] >= min_cohesion[0])
                         and (score[1] >= min_cohesion[1])
                         and (score[2] >= min_count)}

        if not remove_subword:
            return word_to_score

        words = {}

        for word, score in sorted(word_to_score.items(), key=lambda x: len(x[0])):
            len_word = len(word)
            if len_word <= 2:
                words[word] = score
                continue

            try:
                subword = word[:-1]
                subscore = self.get_cohesion_probability(subword)
                droprate = score[2] / subscore[2]

                if (droprate >= min_droprate) and (subword in words):
                    del words[subword]

                words[word] = score

            except:
                print(word, score, subscore)
                break

        return words

    def transform(self, docs, l_word_set):

        def left_match(word):
            for i in reversed(range(1, len(word) + 1)):
                if word[:i] in l_word_set:
                    return word[:i]
            return ''

        return [[left_match(word) for sent in doc.split('  ') for word in sent.split() if left_match(word)] for doc in
                docs]

    def load(self, fname):
        try:
            with open(fname, encoding='utf-8') as f:

                next(f)  # SKIP: parameters(left_min_length left_max_length ...
                token = next(f).split()
                self.left_min_length = int(token[0])
                self.left_max_length = int(token[1])
                self.right_min_length = int(token[2])
                self.right_max_length = int(token[3])

                next(f)  # SKIP: L count
                is_right_side = False

                for line in f:

                    if '# R count' in line:
                        is_right_side = True
                        continue

                    token = line.split('\t')
                    if is_right_side:
                        self.R[token[0]] = int(token[1])
                    else:
                        self.L[token[0]] = int(token[1])

        except Exception as e:
            print(e)

    def save(self, fname):
        try:
            with open(fname, 'w', encoding='utf-8') as f:

                f.write('# parameters(left_min_length left_max_length right_min_length right_max_length)\n')
                f.write('%d %d %d %d\n' % (
                self.left_min_length, self.left_max_length, self.right_min_length, self.right_max_length))

                f.write('# L count')
                for word, freq in self.L.items():
                    f.write('%s\t%d\n' % (word, freq))

                f.write('# R count')
                for word, freq in self.R.items():
                    f.write('%s\t%d\n' % (word, freq))

        except Exception as e:
            print(e)

    def words(self):
        words = set(self.L.keys())
        words = words.union(set(self.R.keys()))
        return words
      
      
class CohesionTokenizer:
    def __init__(self, cohesion):
        self.cohesion = cohesion
        self.range_l = cohesion.left_max_length

    def tokenize(self, sentence, max_ngram=4, length_penalty=-0.05, ngram=False, debug=False):

        def flatten(tokens):
            return [word for token in tokens for word in token]

        tokens = [self._recursive_tokenize(token, max_ngram, length_penalty, ngram, debug) for token in
                  sentence.split()]
        words = flatten(tokens)

        if not debug:
            tokens = [word if type(word) == str else word[0] for word in words]

        return tokens

    def _recursive_tokenize(self, token, max_ngram=4, length_penalty=-0.05, ngram=False, debug=False):

        length = len(token)
        if length <= 2:
            return [token]

        range_l = min(self.range_l, length)

        scores = self._initialize(token, range_l, length)
        if debug:
            pprint(scores)

        result = self._find(scores)

        adds = self._add_inter_subtokens(token, result)

        if result[-1][2] != length:
            adds += self._add_first_subtoken(token, result)

        if result[0][1] != 0:
            adds += self._add_last_subtoken(token, result)

        result = sorted(result + adds, key=lambda x: x[1])

        if ngram:
            result = self._extract_ngram(result, max_ngram, length_penalty)

        return result

    def _initialize(self, token, range_l, length):
        scores = []
        for b in range(0, length - 1):
            for r in range(2, range_l + 1):
                e = b + r

                if e > length:
                    continue

                subtoken = token[b:e]
                score = self.cohesion.get_cohesion_probability(subtoken)
                # (subtoken, begin, end, cohesion_l, frequency_l, range)
                scores.append((subtoken, b, e, score[0], score[2], r))

        return sorted(scores, key=lambda x: (x[3], x[5]), reverse=True)

    def _find(self, scores):
        result = []
        num_iter = 0

        while scores:
            word, b, e, cp_l, freq_l, r = scores.pop(0)
            result.append((word, b, e, cp_l, freq_l, r))

            if not scores:
                break

            removals = []
            for i, (_1, b_, e_, _2, _3, _4) in enumerate(scores):
                if (b_ < e and b < e_) or (b_ < e and e_ > b):
                    removals.append(i)

            for i in reversed(removals):
                del scores[i]

            num_iter += 1
            if num_iter > 100: break

        return sorted(result, key=lambda x: x[1])

    def _add_inter_subtokens(self, token, result):
        adds = []
        for i, base in enumerate(result[:-1]):
            if base[2] == result[i + 1][1]:
                continue

            b = base[2]
            e = result[i + 1][1]
            subtoken = token[b:e]
            adds.append((subtoken, b, e, 0, self.cohesion.L.get(subtoken, 0), e - b))

        return adds

    def _add_first_subtoken(self, token, result):
        b = result[-1][2]
        subtoken = token[b:]
        score = self.cohesion.get_cohesion_probability(subtoken)
        return [(subtoken, b, len(token), score[0], score[2], len(subtoken))]

    def _add_last_subtoken(self, token, result):
        e = result[0][1]
        subtoken = token[0:e]
        score = self.cohesion.get_cohesion_probability(subtoken)
        return [(subtoken, 0, e, score[0], score[2], e)]

    def _extract_ngram(self, words, max_ngram=4, length_penalty=-0.05):

        def ngram_average_score(words):
            words = [word for word in words if len(word) > 1]
            scores = [word[3] for word in words]
            return max(0, np.mean(scores) + length_penalty * len(scores))

        length = len(words)
        scores = []

        if length <= 1:
            return words

        for word in words:
            scores.append(word)

        for b in range(0, length - 1):
            for r in range(2, max_ngram + 1):
                e = b + r

                if e > length:
                    continue

                ngram = words[b:e]
                ngram_str = ''.join([word[0] for word in ngram])
                ngram_str_ = '-'.join([word[0] for word in ngram])

                ngram_freq = self.cohesion.L.get(ngram_str, 0)
                if ngram_freq == 0:
                    continue

                base_freq = min([word[4] for word in ngram])
                ngram_score = np.power(ngram_freq / base_freq, 1 / (r - 1)) if base_freq > 0 else 0
                ngram_score -= r * length_penalty

                scores.append((ngram_str_, words[b][1], words[e - 1][2], ngram_score, ngram_freq, 0))

        scores = sorted(scores, key=lambda x: x[3], reverse=True)
        return self._find(scores)