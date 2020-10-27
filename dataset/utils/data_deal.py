"""
@Description: 加载数据，保存数据
@Author: Bao Wenjie
@Email: bwj_678@qq.com
@Date: 2020/10/27
"""
import csv
class DataDeal():
    def __init__(self):
        super().__init__()
    
    @classmethod
    def load_data(self, path, cls='\t', encoding="utf-8-sig", quotechar=None):
        """ 从文件读取数据\n
        @param:\n
        :path: 路径\n
        :cls: 分隔符（一行内）\n
        :encoding: 编码方式\n
        :quotechar: 引用符\n
        @return:\n
        :lines: list(),读取的数据\n
        """
        with open(path, "r", encoding=encoding) as f:
            reader = csv.reader(f, delimiter=cls, quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
        return lines


if __name__ == "__main__":
    data = DataDeal.load_data(path='dataset/CONLL2003/test.txt', cls=' ')
    print(data[:9])