# Inpainting_diffusion(这是我关于基于扩散重建的异常检测的一些探索)
有监督缺陷检测的问题：1、获取缺陷数据困难；2、缺陷标注困难；3、缺陷间差异巨大；4、缺陷与背景差异小；5、缺陷形式、种类、大小未知。故我们通过无监督的缺陷检测算法进行识别，而无监督检测算法主要分为基于特征嵌入以及基于重建的方法，其中基于特征嵌入的方法多需要预训练模型，不易在工业实际的应用，故我通过研究基于重建的方法进行，重建多采用生成模型。
##1、扩散模型与异常检测
扩散模型：和其他生成模型一样，实现从噪声（采样的简单分布）生成目标数据样本。扩散模型包括两个过程：前向过程和反向过程，其中，无论是前向过程还是反向过程都是一个参数化的马尔可夫链，我们利用反向过程生成数据样本（它的作用类似GAN中的生成器以及AE，只不过AE是一步的且会有维度变化，而DDPM的反向过程没有维度变化）。
![扩散模型](https://github.com/xiaokai580/Inpainting_diffusion/assets/82256486/520b8740-90a7-45f9-b528-e09d8a581fed)
##2、基于扩散重建面临的问题
扩散模型重建时，需要逐步的前向加噪声，把原来正常的图像变为随机的高斯噪声图片。而我们基于扩散进行缺陷检测时，不希望逐步加噪声到高斯噪声，因为这个过程是较长的，在工业部署上是不切实际的；同时，通过加入太多噪声，会导致随机性变大，使原是正常图像重建为缺陷图像。故我们多数让其前向加噪一定步数后进行重建，我们可以从下图可见，随着加噪步数的增大，缺陷被噪声掩盖的越多，直到消失。同时，不同步数下重建后的图像是有区别的，其与原图的相似性是与迭代的噪声成反比的，得到的缺陷结果是不一样的，故如何控制加噪步数是基于扩散重建所面临的重要问题。
![扩散问题](https://github.com/xiaokai580/Inpainting_diffusion/assets/82256486/d8a6cd4c-6d58-4faa-b6dc-4e29a263c76b)
##3、重建模型设计
针对以上的问题，我设计了一个基于自编码器与扩散模型相结合的重建模型，其模型结构如下图所示：
模型整体结构图：
![模型结构图](https://github.com/xiaokai580/Inpainting_diffusion/assets/82256486/95d32c2d-2f58-46bd-a70f-6cf0788b9123)
去噪模型结构：
![去噪模型结构图](https://github.com/xiaokai580/Inpainting_diffusion/assets/82256486/a3276b2e-4b35-48a1-8631-065eb81060aa)
时间编码结构：
![时间编码结构](https://github.com/xiaokai580/Inpainting_diffusion/assets/82256486/84888168-0c46-4777-b69b-4a52ae4f8da9)
首先，利用自编码模型，把原图有大缺陷的进行重建，之后利用扩散模型进行小步去噪优化重建效果。