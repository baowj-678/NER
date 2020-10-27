import sys
sys.path.append("..")
from utils import DataDeal

class Conll2003Reader():
    def __init__(self, path):
        super().__init__()
    
    @classmethod
    def load_data(self, path):
        """ 加载Conll2003数据集
        @param:
        :path: 文件路径
        @return:
        :data list(sent, name-entity): 语句和命名实体
        """
        tmp = DataDeal.load_data(path, cls=' ')
        data = []
        sentence = []
        named_entity = []
        for line in tmp:
            if len(line) < 1:
                if len(sentence) > 0:
                    data.append([sentence, named_entity])
                sentence = []
                named_entity = []
            else:
                if line[0] == '-DOCSTART-':
                    continue
                else:
                    sentence.append(line[0])
                    named_entity.append(line[-1])
        return data



if __name__ == "__main__":
    data = Conll2003Reader.load_data('test.txt')
    sent = data[:7]
    for line in sent:
        print(' '.join(line[0]),'\n',  ' '.join(line[1]))
