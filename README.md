# TensorQ
一个基于张量网络的大规模量子线路模拟工具包

# 安装

cd <yourpath>TensorQ-main

python setup.py install

# 依赖

artensor==0.1.0

你可以从我的仓库下载0.1.0版本的artensor（推荐）

你也可以下载最新的artensor，[Fanerst/artensor: Generating contraction orders and perform numerical contractions for arbitrary tensor networks (github.com)](https://github.com/Fanerst/artensor)

 cirq==0.7.0

# 原理
 
 artensor和tensorq采用张量网络解决指数墙问题的原理，以及张量网络的切片方法可以参考这篇文献：
 
 F. Pan, K. Chen, and P. Zhang, Solving the sampling problem of the sycamore quantum circuits, [Phys. Rev. Lett. 129, 090502 (2022).](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.090502)
