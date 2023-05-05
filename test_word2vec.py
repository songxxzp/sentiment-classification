from gensim.models import Word2Vec, KeyedVectors
import torch.nn.functional as F

word2vec_model = KeyedVectors.load_word2vec_format('./Dataset/wiki_word2vec_50.bin', binary=True)

print(word2vec_model)

print(word2vec_model.vector_size)
print(len(word2vec_model.index_to_key))

for i, k in enumerate(word2vec_model.index_to_key):
    print(i, k)

print(word2vec_model.get_vector("你"))

print(word2vec_model.get_index("你"))
