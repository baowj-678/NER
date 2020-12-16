# NER:rocket:









### 注意事项:exclamation:

* 调用兄弟目录下**Package**：

    ~~~python
    import sys
    import os
    abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(abs_path)
    ~~~

* **\_\_init\_\_.py**文件添加：

    ~~~python
    import sys
    import os
    abs_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(abs_path)
    ~~~

    