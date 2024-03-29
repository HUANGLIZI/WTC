# Semi-WTC
This repository includes the official project of Semi-WTC Model, presented in our paper: Semi-WTC: A Practical Semi-supervised Framework for Attack Categorization through Weight-Task Consistency (Under Major Revision)

![image](https://github.com/HUANGLIZI/WTC/blob/main/img/SEMI-WTC.jpg)

*Paper Link*: https://arxiv.org/abs/2205.09669

*Email*: zihanli@stu.xmu.edu.cn

Please contact me if you need the dataset.

# Usage

dataset/ : the used dataset for the model

utils/ : all the supplementary tools used for our Semi-WTC

you can use the command "python AAR-NSLKDD.py" for ACTIVE ADAPTION RESAMPLING (AAR).

you can use the command "python main_demo.py" for demo training and testing.

At present, the demo file "main_demo.py" is under preparation.

# Environment

Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

# Experimental results for NSL-KDD demo

Acc & F1:

![image](https://github.com/HUANGLIZI/WTC/blob/main/img/results.jpg)

Confusion Matrix:

![image](https://github.com/HUANGLIZI/WTC/blob/main/img/CM.jpg)

# Citation

```bash
@article{li2022semi,
  title={Semi-WTC: A Practical Semi-supervised Framework for Attack Categorization through Weight-Task Consistency},
  author={Li, Zihan and Chen, Wentao and Wei, Zhiqing and Luo, Xingqi and Su, Bing},
  journal={arXiv preprint arXiv:2205.09669},
  year={2022}
}
```
