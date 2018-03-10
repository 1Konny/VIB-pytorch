# Deep Variational Information Bottleneck
<br>

### Overview
Pytorch implementation of Deep Variational Information Bottleneck([paper], [original code])

![ELBO](misc/ELBO.PNG)
![monte_carlo](misc/monte_carlo.PNG)
<br>

### Dependencies
```
python 3.6.4
pytorch 0.3.1.post2
tensorboardX(optional)
tensorflow(optional)
```
<br>

### Usage
```
python main.py --beta 1e-3 --num_avg 12
```
<br>

### Results
<br>

### References
1. Deep Variational Information Bottleneck, Alemi et al.
2. Tensorflow Demo : https://github.com/alexalemi/vib_demo

[paper]: http://arxiv.org/abs/1612.00410
[original code]: https://github.com/alexalemi/vib_demo
