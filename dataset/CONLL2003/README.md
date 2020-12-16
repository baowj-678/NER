# 数据说明

**来源**：[http://aclweb.org/anthology/W03-0419](http://aclweb.org/anthology/W03-0419)

| 单词word | 词性second a part-of-speech (POS) | 语法块syntactic chunk | 实体标签named entity |
| -------- | --------------------------------- | --------------------- | -------------------- |
| TAKE     | NNP                               | I-NP                  | O                    |

* **I-TYPE**：which means that the word is inside a phrase of type TYPE
* **B-TYPE**：Only if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag B-TYPE to show that it starts a new phrase
* **O**：A word with tag O is not part of a phrase

| 标签   | 详情       |
| ------ | ---------- |
| O      | **NULL**   |
| B-LOC  | 新**地址** |
| I-LOC  | **地址**   |
| B-PER  | 新**人名** |
| I-PER  | **人名**   |
| B-MISC | 新         |
| I-MISC |            |
| B-ORG  | 新**组织** |
| I-ORG  | **组织**   |

