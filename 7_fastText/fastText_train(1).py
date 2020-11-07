from gensim.models import FastText
from gensim.models.word2vec import PathLineSentences

sentences = PathLineSentences("./data/1billion/")

# set 1 : Model(CBOW) n-gram(2~3) Negative Sampling(15), 17:00 ~ 20:00 (3h)
# model = FastText(sentences=sentences, size=100, window=5, min_count=10, workers=4, sg=0, hs=0,
#                   negative=15, ns_exponent=0.75, alpha=0.01, min_alpha=0.0001, iter=5,
#                  word_ngrams=1, min_n=2, max_n=3)
# model.save("fastText1.model")
# print(len(model.wv.vocab))
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)

# set 2 : Model(Skipgram) n-gram(3~6) : 20:40 ~ 04:00 (8h)
# model = FastText(sentences=sentences, size=100, window=5, min_count=10, workers=4, sg=1, hs=0,
#                   negative=15, ns_exponent=0.75, alpha=0.01, min_alpha=0.0001, iter=5,
#                  word_ngrams=1, min_n=3, max_n=6)
# model.save("fastText2.model")
# print(len(model.wv.vocab))
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)

# set 3 : Dimension 300 으로할떄 측정, size up , iter down : 10:30 ~ 13:30 (3h)
# model = FastText(sentences=sentences, size=300, window=5, min_count=10, workers=4, sg=0, hs=0,
#                   negative=15, ns_exponent=0.75, alpha=0.01, min_alpha=0.0001, iter=3,
#                  word_ngrams=1, min_n=2, max_n=3)
# model.save("fastText3.model")
# print(len(model.wv.vocab))
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)

# set 4 : 15:00 ~ 18:00 (3h)
# model = FastText(sentences=sentences, size=150, window=3, min_count=10, workers=4, sg=0, hs=0,
#                   negative=15, ns_exponent=0.75, alpha=0.01, min_alpha=0.0001, iter=3,
#                  word_ngrams=1, min_n=3, max_n=6)
# model.save("fastText4.model")
# print(len(model.wv.vocab))
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)

# evalutation
model = FastText.load("fastText4.model")
score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
print(score)
print(model.wv.most_similar("lemon____ade", topn=20))
print(len(model.wv.vocab))


