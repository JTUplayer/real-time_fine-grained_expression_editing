# 实时细粒度表情编辑
该项目支持实时的细粒度表情编辑工作，使用PySimpleGUI搭建了一个用于实时交互的用户界面，在NVIDIA GeForce RTX 3090下的运行效率能够达到每秒27次编辑，视觉上表现为
表情的连续编辑
## 效果展示
交互界面展示

![交互界面](https://github.com/JTUplayer/real-time_fine-grained_expression_editing/blob/master/imgs/GUI0.PNG) 

编辑结果展示
![编辑效果](https://github.com/JTUplayer/real-time_fine-grained_expression_editing/blob/master/imgs/GUI1.PNG) 


##使用步骤
（1）需要首先下载StyleGAN2的官方模型，放置主目录下

（2）若需要编辑自己的真实图片，首先将图片裁剪到与CelebA-HQ数据类似的格式（人脸居中，且占大部分内容），再使用
pSp或者Image2StyleGAN完成图像的逆映射，获得隐编码放入latent/latents.npy（nx18x512）

（3）运行main.py
