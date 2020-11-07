from gensim.models.keyedvectors import KeyedVectors
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences

#-------------pre-trained word2vec---------------
# model = KeyedVectors.load_word2vec_format("./data/GoogleNews-vectors-negative300.bin", binary=True, limit=60000)
# score, predictions = model.evaluate_word_analogies('./data/questions-words.txt')

# print(score)
# print(model['apple'])
# print("similarity between apple and fruit: {}".format(model.similarity("apple", "fruit")))
# print("similarity between apple and car: {}".format(model.similarity("apple", "car")))
# print(model.most_similar("apple", topn=10))
# print(model.most_similar(positive=['king', 'women'], negative=['man'], topn=10))

#-------------training---------------
#set0
# sentences = PathLineSentences("./data/1billion/")
# model = Word2Vec(sentences, size=30, window=5, min_count=5, workers=4, sg=0, hs=1,
#                  negative=5, ns_exponent=0.80, cbow_mean=1, alpha=0.01, min_alpha=0.0001, iter=1)
# model.save("word2vec.model")
# print(len(model.wv.vocab))
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)

#set1
# sentences = PathLineSentences("./data/1billion/")
# model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=4, sg=0, hs=1,
#                  negative=15, ns_exponent=0.80, cbow_mean=1, alpha=0.01, min_alpha=0.0001, iter=5)
# model.save("word2vec1.model")
# print(len(model.wv.vocab))
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)

#set2
# sentences = PathLineSentences("./data/1billion/")
# model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=4, sg=1, hs=1,
#                  negative=15, ns_exponent=0.80, cbow_mean=1, alpha=0.01, min_alpha=0.0001, iter=5)
# model.save("word2vec3.model")
# print(len(model.wv.vocab))
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)

#set3 (iter3)
# sentences = PathLineSentences("./data/1billion/")
# model = Word2Vec(sentences, size=300, window=5, min_count=5, workers=4, sg=0, hs=1,
#                  negative=0, ns_exponent=0.80, cbow_mean=1, alpha=0.01, min_alpha=0.0001, iter=3)
# model.save("word2vec3.model")
# print(len(model.wv.vocab))
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)

#set4
# sentences = PathLineSentences("./data/1billion/")
# model = Word2Vec(sentences, size=300, window=3, min_count=5, workers=4, sg=1, hs=1,
#                  negative=0, ns_exponent=0.80, cbow_mean=1, alpha=0.01, min_alpha=0.0001, iter=3)
# model.save("word2vec4.model")
# print(len(model.wv.vocab))
# score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
# print(score)

#-------------evaluation---------------
model = Word2Vec.load("word2vec4.model")
score, predictions = model.wv.evaluate_word_analogies('./data/questions-words.txt')
print(score)

print(model.wv.most_similar("car", topn=200))
print(len(model.wv.vocab))
print("similarity between apple and fruit: {}".format(model.wv.similarity("apple", "fruit")))
print(model.wv["apple"])


