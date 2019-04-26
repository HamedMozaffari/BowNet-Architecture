# BowNet-Architecture
Codes for the BowNet and wBowNet models
BowNet and wBowNet are two new FCN (Fully Convolutional Networks) which use dilation convolutial layers as well as standard convolutional layers. Their performance and accuracy is better than or similar to the famous UNET architecture although their parameters are smaller significantly. 
Architectures of these two models can be seen below: 

# BowNet 
This model concatenates two forward path feature maps to find a better segmenation. One path contains standard convolutions and the other contains dilated convolutions. Because those two path have not any collaboration, results of those two are separate and independent. 
![BowNet2](https://user-images.githubusercontent.com/34034638/56823446-dc417e80-6821-11e9-81c8-4a525f82ed78.png)

# wBowNet
wBowNet is an inteconected model of BowNet which two paths are dependent, and there is a collaboration between dilated convolution and standard onces. 
![wBowNet](https://user-images.githubusercontent.com/34034638/56823448-dea3d880-6821-11e9-82f3-d3d56ad6739a.png)

Name of the architecture is BowNet because of mimicing its shape with a Bow. 
As one can see, the number of layers and kernels are smaller than VGG16 network; however, our results over several benchmarks revealed that BowNet architecture can be an alternative method for applications with high performance demand while high accuracy is favourite as well. 
