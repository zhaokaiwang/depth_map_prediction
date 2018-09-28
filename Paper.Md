# Deeper Monocular Depth Prediction via Fully Convolutional DenseNets

##Abstarct

## 1.Introduction

1 引出深度估计这个问题，说明问题是有意义的

单目图像深度估计是计算机视觉和图像处理领域最基本的问题之一，同时这也是一个非常具有挑战性的工作，也是目前视觉方面一个非常活跃的领域。传统方法解决这个问题最成功的例子之一就是 Structure-from-Motion(SfM)[23],它通过相邻两帧之间的相机运动来得到深度信息。除了运动信息之外，亮度信息[24]也被用来得到深度信息。

在缺少了上述的条件的情况下，从一张色彩图片中恢复深度信息是一个非常困难的工作，这是因为它ill-posed问题的本质。即使对我们的大脑来说，从色彩图片中得到深度信息也是一件非常困难的事，我们通常只能猜测到非常宽泛的距离，这也是在我们有一定的先验知识下面。单目图像深度估计不仅仅帮助我们能在没有深度传感器的条件下获得深度信息，同时，获得精确的深度信息后，它通常也能帮助其他的视觉问题比如语义分割[4, 5], 平面法线估计(surfact normal)[5], 帧间运动[20, 15]等等。

2 介绍最近的工作，准备引出自己的方法

最近，基于卷积网络的（Convolutional Neural Networks）state-of-the-art的单目图像深度估计方法逐渐成为主流[3, 4, 5, 6, 7, 8, 14],
CNNs的方法经常结合条件随机场(CRFs Conditional Random Fields)作为后处理(Post-precessing)方法[6],或者通过随机森林(Random Forset)[18],这些方法都非常的复杂，需要大量的参数和数据驱动。与此同时，这些卷积神经网络CNNs的方法在准确性上有非常的提升在标准的数据集上，是目前最好的方法。

最近，一个新的卷积神经网络结构--DenseNet[1].DenseNet由Dense Block和Transition layer组成，Dense block是一种迭代拼接前面学习到的feature map的一种结构，Transition layer通过池化操作完成。这个新的网络结构可以看成对一种对ResNets的延伸，ResNets直接通过加法的方式将前面学到的特征图传到后面的层，而DenseNet是将前面学习到的特征图直接拼接到后面的层，实现了特征的复用。虽然这是一个小的改变，但是它带来了以下几个好处：(1)节省参数，相比于ResNets，DenseNet在相同层数下节省了更多的参数。(2)暗含的高层次的信息，底层的特征通过拼接直接传到后面。(3)特征复用，所有的层都可以感受到前面所有的层学习到特征。这些属性使得DenseNets适合做深度估计，因为它的跳连接(skip connecetion)的设计和结合的多个尺度的特征信息的本质。



3 简单介绍自己的工作

在这篇论文中，我们扩展了通过一种通过Dense net进行语义分割的网络结构[11]，因为深度估计和语义分割的相似性，我们只修改了像素上升路径中(Upsampling path)的结构，通过结合不同和分辨率下的输出，我们可以的到更好的输出。和[11]相似，在上升路径中，我们只上采样上一个dense Block学习到的特征图


4 总结自己的工作， 

## 2.Related Work

## 3.Approach

## 4.Experimental Results

## 5.Conclusion

## 6.References