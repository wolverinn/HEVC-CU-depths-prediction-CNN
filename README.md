# HEVC-CU-depths-prediction-CNN
Using convolutional neural networks to predict the Coding Units (CUs) depths in HEVC intra-prediction mode. Achieve a trade-off between the encoding time in HEVC and the BDBR.

## Introduction
In HEVC intra-prediction mode, it takes the HEVC encoder a lot of time to decide the best depth for CTUs. So we try to use CNN to predict the CTU depth decision, and try to achieve a trade-off between the encoding time and the BDBR.

For a 64x64 CTU, the HEVC encoder predicts a 16x16 matrix to represent its depth decision. We can further process this 16x16 matrix and extract 16 labels to represent the depth prediction for a 64x64 CTU (because elements in each 4x4 block in this matrix is identical). So in short, we can use a 64x64 CTU (which is an image) as the input of our network, and output a vector of length 16.

Directly predict 16 labels at one time is difficult for a neural network, so there're other ways to design the model. A 64x64 CTU corresponds to 16 labels. And if we divide it into four 32x32 CUs, then each 32x32 CU corresponds to only four labels, which makes the task easier.

The depths are 0, 1, 2 or 3. Depth 0 indicates that the 64x64 CU will be encoded as it is. Depth 1 indicates that the 64x64 CU will be further split into four 32x32 CUs and then be encoded, etc. Here's an example of a 64x64 CU and its depth decision:

![CU depths](_v_images/20191116214742584_27076.png)

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
The **loss** of our trained model on test set is: 3.1049

The **accuracy** of each label predicted on test set is: 66.12%

The best way to evaluate the model is to integrate the model into the HEVC encoder. I've realized a pipeline, see it in:

[HEVC-deep-learning-pipeline](https://github.com/wolverinn/HEVC-deep-learning-pipeline)

Using this evaluating pipeline, we can compare the change in encoding time and BDBR at the same time.

|     YUV sequence      | bitrate-HEVC | bitrate-CNN | Y-PSNR-HEVC | Y-PSNR-CNN | YUV-PSNR-HEVC | YUV-PSNR-CNN | Time-HEVC | Time-CNN |
| --------------------- | ------------ | ----------- | ----------- | ---------- | ------------- | ------------ | --------- | -------- |
| Flowervase_416x240_30 | 522.3700     | 556.9627    | 34.0869     | 33.8087    | 35.2319       | 34.9543      | 1225.481  | 367.107  |
| Keiba_416x240_30      | 539.8007     | 582.1690    | 33.4852     | 33.3019    | 34.5684       | 34.3756      | 1281.129  | 356.524  |
| mobile_352x288_20     | 1853.5339    | 1955.5392   | 28.2047     | 27.9656    | 29.3071       | 29.0889      | 1629.696  | 666.098  |
| RaceHorses_832x480_30 | 2594.1504    | 2719.9232   | 32.1133     | 31.9442    | 32.9179       | 32.7702      | 5440.564  | 1513.008 |
| ParkScene_1920x1080_24 | 7361.9354    | 7702.8979   | 32.7847     | 32.5906    | 33.9392       | 33.7562      | 13238.985 | 3591.274 |

The average **BD bitrate** increase is: 6.60%. The average loss in **BD-PSNR** is: 0.6dB. The average **encoding time** reduction is: 71.5%.

Another way to evaluate the model is to compare the difference in RD-cost for each YUV file encoded. As ```xCompressCU()``` in HEVC encoder calculates the RD-cost exhaustively at each depth, we can get the RD-cost for every possible depth decision. Thus, we can realize comparison of RD-cost between the original encoder and the CNN model. See the ```test RD-cost``` folder for codes.

The increase in **RD cost** of our model is: 2.1% (tested only on one YUV sequence)

## To be continued...
Since 1 label comes from a 16x16 CU, so we can simply predict 1 label at a time. The input can be a combination of 64x64, 32x32 and 16x16 CUs. I think this will achieve higher accuracy... Also, some pre-trained models like ResNet can be tried...

If you find this repository useful, please give it a star! :heart:
