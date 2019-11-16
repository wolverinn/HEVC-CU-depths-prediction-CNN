# HEVC-CU-depths-prediction-CNN
Using convolutional neural networks to predict the Coding Units (CUs) depths in HEVC intra-prediction mode. Achieve a trade-off between the encoding time in HEVC and the BDBR.

## Introduction
In HEVC intra-prediction mode, it takes the HEVC encoder a lot of time to decide the best depth for CTUs. So we try to use CNN to predict the CTU depth decision, and try to achieve a trade-off between the encoding time and the BDBR.

For a 64x64 CTU, the HEVC encoder predicts a 16x16 matrix to represent its depth decision. We can further process this 16x16 matrix and extract 16 labels to represent the depth prediction for a 64x64 CTU (because each 4x4 block in this matrix is identical). So in short, we can use a 64x64 CTU (which is an image) as the input of our network, and output a vector of length 16.

Directly predict 16 labels at one time is difficult for a neural network, so there're other ways to design the model. A 64x64 CTU corresponds to 16 labels. And if we divide it into four 32x32 CUs, then each 32x32 CU corresponds to only four labels, which makes the task easier.

The depths are 0, 1, 2 or 3. Depth 0 indicates that the 64x64 CU will be encoded as it is. Depth 1 indicates that the 64x64 CU will be further split into four 32x32 CUs and then be encoded, etc. Here's an example of a 64x64 CU and its depth decision:

![64CU](_v_images/20191116110752638_8757.png =425x)

![](_v_images/20191116110808920_18707.png =425x)

For more information on a CNN approach to predict the CU depths for a 64x64 CTU, you can refer to these documents:

- [A deep convolutional neural network approach for complexity reduction on intra-mode HEVC](https://ieeexplore.ieee.org/document/8019316)
- [Fast CU Depth Decision for HEVC Using Neural Networks](https://ieeexplore.ieee.org/document/8361836)

## CNN model
Like mentioned above, we can directly use a 32x32 CU as input, and ouput 4 labels. But if we know how the depth 0/1/2/3 is decided, then this model doesn't make sense for depth-0, because a 32x32 CU is only part of a 64x64 CU, and it won't be sufficient to decide whether the 64x64 CU should be split or non-split.

So our model use both the 64x64 CU and the current 32x32 CU as input, and ouputs 4 labels indicating depths. Here's our architecture:

![cnn_model](_v_images/20191116195804171_10635.png)

## Dataset & Loss Function
We generate our own dataset from YUV test sequences, refer to:

[HEVC-CU-depths-dataset](https://github.com/wolverinn/HEVC-CU-depths-dataset)

We use Cross Entropy Loss as loss function. For the four output labels, we calculate the Cross Entropy Loss seperately and then add them together.

## Validation
The loss of our trained model is: 

The accuracy of each label predicted is: 

