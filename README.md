### Dataset

https://hkustconnect-my.sharepoint.com/:f:/g/personal/yruanaf_connect_ust_hk/EqliD2zZ4X9CiUPbovRfSn8Ba47Bd1tLtiCtjBCCFZzxXg?e=BwoHsv

### Main structure from

[kaize0409/GPN_Graph-Few-shot: Implementation of CIKM2020 -- Graph Prototypical Networks for Few-shot Learning on Attributed Networks (github.com)](https://github.com/kaize0409/GPN_Graph-Few-shot)

### Models implementation from

[colflip/gnns_fewshot: code implementation of GNNs in few-shot learning: GCN, GAT, GraphSAGE to the node classification task of some datasets. (github.com)](https://github.com/colflip/gnns_fewshot)

Note: for this, the following dependencies are required

- Python ≥ 3.10
- PyTorch ≥ 11.3
- pyg ≥ 1.12.0

My settings:

- Python 3.9

- PyTorch    1.10.1+cpu

- for pyg, I follow the following link [(74条消息) pytorch正确的安装torch_geometric,无bug、多种类版本_模糊包的博客-CSDN博客_安装torch版本](https://blog.csdn.net/xinjieyuan/article/details/120483494)

- Used the below command to solve some bugs (forgot the details.. )

  ​    pip3 install --upgrade protobuf==3.20.1

### Train a model

sh train.sh

- Model: specify in *train.sh* 
- Datasets: specify in *main_train.py*

### Result table

[results.xlsx](https://hkustconnect-my.sharepoint.com/:x:/g/personal/yruanaf_connect_ust_hk/EZH4ct9QFlNNif4KfH6FLMEBzPl3C4kzhsjAF1H3D1i24Q?e=LnCrEI)

