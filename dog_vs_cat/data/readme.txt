数据加载依靠dataset dataloader这两个类来实现
关于数据加载的相关操作，其基本原理就是使用Dataset进行数据集的封装，再使用Dataloader实现数据并行加载。
对于训练集，我们希望做一些数据增强处理，如随机裁剪、随机翻转、加噪声等，而验证集和测试集则不需要。

关于数据：
训练数据以及测试数据放在~/Data/lp/dog_vs_cat下
其中train val在sample文件夹下
test即为test文件夹
