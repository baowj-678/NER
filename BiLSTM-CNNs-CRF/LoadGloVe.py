from typing import Optional, Union
from scipy.stats.stats import mode
import torch
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def load_glove(glove_file: Optional[str], word2vec_file: str, vocab, embed_size: int) -> torch:
    """ 导入glove数据到pytorch\n
    @param:\n
    :glove_file: GloVe文件路径(如果为None,使用word2vec_file;否则转成word2vec存放在word2vec_file).\n
    :word2vec_file: word2vec文件路径\n
    :vocab: 字典.\n
    :embed_size: 词向量维度.\n
    @return:\n
    :embedding :torch.Embedding\n
    """
    # glove -> word2vec
    if glove_file is not None:
        print(glove_file)
        glove2word2vec(glove_file, word2vec_file)
    # 加载模型
    model = KeyedVectors.load_word2vec_format(word2vec_file)
    # load data
    vocab_size = len(vocab) + 1
    weight = torch.zeros(vocab_size + 1, embed_size)
    for word in model.index2word:
        index = vocab[word]
        if index != vocab._pad_i_:
            weight[index, :] = torch.from_numpy(model.get_vector(word))

    # 生成Embedding
    embedding = torch.nn.Embedding.from_pretrained(weight)

    return embedding

if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format('D:/NLP/Word2Vec/GloVe/glove6B50d.txt')