import sys
import os
abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abs_path)
from utils import DataDeal

class Conll2003Processor():
    def __init__(self, path):
        super().__init__()
    
    @staticmethod
    def make_vocab(src_path, word_index, entity_index, save_path):
        """ 加载Conll2003数据集
        @param:
        :src_path: (list/str) 文件路径
        @return:
        :data list(sent, name-entity): 语句和命名实体
        """
        vocab = {}
        entitys = {}
        def make_vocab_single(path):
            """ 单文件处理 """
            data = DataDeal.load_data(path=path, delimiter=' ')
            for line in data:
                if len(line) > 1:
                    word = line[word_index]
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
                    entity = line[entity_index]
                    if entity in entitys:
                        entitys[entity] += 1
                    else:
                        entitys[entity] = 1

        if isinstance(src_path, list):
            for path in src_path:
                make_vocab_single(path)
        else:
            make_vocab_single(src_path)
        vocab_save_path = os.path.join(save_path, 'vocab.txt')
        vocab_analyse_save_path = os.path.join(save_path, 'vocab_analyse.json')
        entity_save_path = os.path.join(save_path, 'entity.txt')
        entity_analyse_save_path = os.path.join(save_path, 'entity_analyse.json')
        DataDeal.save_dict_json(vocab_analyse_save_path, vocab)
        DataDeal.save_dict_json(entity_analyse_save_path, entitys)
        DataDeal.save_single(vocab_save_path, list(vocab.keys()))
        DataDeal.save_single(entity_save_path, list(entitys.keys()))
        return


if __name__ == "__main__":
    src_path = ['dataset/CONLL2003/test.txt', 'dataset/CONLL2003/train.txt', 'dataset/CONLL2003/valid.txt']
    save_path = 'dataset/CONLL2003/'
    Conll2003Processor.make_vocab(src_path, 0, -1, save_path)