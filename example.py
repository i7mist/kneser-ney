from __future__ import print_function

from nltk.corpus import gutenberg
from nltk.util import ngrams
from kneser_ney import KneserNeyLM

gut_ngrams = (
    ngram for sent in gutenberg.sents() for ngram in ngrams(sent, 3,
    pad_left=True, pad_right=True))
lm = KneserNeyLM(3, gut_ngrams)
# print(lm.score_sent(('This', 'is', 'a', 'sample', 'sentence', '.')))
print(lm.logprob2(('By', 'Jane', 'Austen')))
print(lm.logprob2(('By', 'Jane')))
print(lm.logprob2(('By',)))
print(lm.logprob2(('', 'By', 'Jane')))
print(lm.logprob2(('', '', 'By')))
