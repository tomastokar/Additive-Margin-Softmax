# Additive-Margin-Softmax
Pytorch implementation of additive margin softmax loss [1].

```
loss = AMSloss(embedding_dim, no_classes, scale = 30.0, margin=0.4)
err = loss(x, labels)
err.backward()
```

where:
  - `embedding_dim` - embedding vector dimension
  - `no_classes` - number of classes to be embedded
  - `scale` - scale factor
  - `margin` - additive margin 

## Demo

A simple demo using Fashio-MNIST dataset [2] can be run by:

```
python demo.py
```

### Results
![AMSloss](results/AMS.png?raw=true "Sphere Plot - AMSloss")

[1] "Additive Margin Softmax for Face Verification." Wang, Feng, Jian Cheng, Weiyang Liu and Haijun Liu. IEEE Signal Processing Letters 25 (2018): 926-930.

[2] "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms." Han Xiao, Kashif Rasul, Roland Vollgraf. arXiv:1708.07747