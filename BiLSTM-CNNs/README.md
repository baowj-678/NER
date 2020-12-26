# End-to-end Sequence Labeling via Bi-directional LSTM-CNNs

## Network

### CNN for Character-level Representation



<img src="md_pic\CNN.png" alt="image-20201022200229254" style="zoom:80%;" />

* **INPUT**：$\bold{I}\in\mathbb{R}^{length\times embedding\_size}$：一个句子每个单词*Embed*成一个*hidden*；

* **dropout**：在**character-embedding**送入**CNN**之前；
* **Conv**：**Kernel_Size**：$n\times1$，**Padding_Size**：$k$；$\mathbb{R}^{length\times embedding\_size}\to \mathbb{R}^{length+2*k-n+1\times embedding\_size}$
* **Max_Pooling**：对*length*进行*pooling*，$\mathbb{R}^{length+2*k-n+1\times embeddig\_size}\to \mathbb{R}^{embedding\_size}$
* **OUTPUT**：$\bold{O}\in\mathbb{R}^{embedding\_size}$

****

### Bi-directional LSTM

**略**



****

#### practice

**参数**：

* $W:(class\_num, class\_num, hidden\_size)$
* $\bold{b}:(class\_num, class\_num)$
* $class\_num$：状态数

***

**过程**

* **INPUT**：$\bold{z}=[z_1, z_2,\cdots,z_n],z_i\in \mathbb{R}^{hidden\_size}$，**BiLSTM**的输出；
* **单步**：$\psi_i=\exp(\bold{W}\cdot \bold{z}_i+\bold{b}),\psi_i\in\mathbb{R}^{class\_num, class\_num}$
* **P**：$P=\sum(\psi_1[y_0]\cdot \psi_1\cdots \psi_n),P\in\mathbb{R}^{1}$

* **P_star**：$P_{star}=\psi_1[y_0,y_1]\cdot \psi_2[y_1,y_2]\cdots \psi_n[y_{n-1},y_n],P_{star}\in\mathbb{R}^1$
* **LikeLihood**：$L=\frac{P_{star}}{P}$

~~~

~~~

