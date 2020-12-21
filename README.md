# SimpleCNNbyCPP
For Course CS205 'C/C++ Program Design' at Southern University of Scicence and Technology, China.

## Model Information
The model is trained to perform face classification (face or background).

Detailed definition: [model.py](./model.py). Visualization: [netron](https://netron.app/?url=https://raw.githubusercontent.com/ShiqiYu/SimpleCNNbyCPP/main/weights/face_binary_cls.onnx) (***NOTE***: you need an extra softmax layer in the end of the pipepline to output scores in the range [0.0, 1.0]).

More about `face_binary_cls.cpp`:
- This file is ported from `face_binary_cls.pth` using `port2cpp` defined in [model.py](./model.py#L123-L203).
- Input: a tensor,
    - loaded from an 128x128 RGB image as ***RGB format*** and ***shape [channel, height, width]***,
    - ***normalized in the range [0.0, 1.0]***.
- Output: a tensor of ***shape [2]***. Softmax is needed to compute confidences in the range [0.0, 1.0]. Values at index 0 stands for the confidence of background, while index 1 for face's.
- Note that the parameters of batch normalization is already combined to convolutional layers' when porting weights (`.pth`) to `.cpp`.

## Examples of locating weights by indexing
A convolutional layer (conv) is defined as `[out_channels, in_channels, kernel_size_h, kernel_size_w]`. It takes a tensor of shape `[in_channels, in_h, in_w]` as input, and ouputs a tensor of shape `[out_channels, out_h, out_w]`. Example of locating weights and bias for a 3x3 kernel at `out_channels=o, in_channels=i`:
```cpp
for (int o = 0; o < out_channels; ++o) {
    for (int i = 0; i < in_channels; ++i) {
        // weights
        // first row of the kernel
        float kernel_oi_00 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 0];
        float kernel_oi_01 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 1];
        float kernel_oi_02 = conv0_weight[o*(in_channels*3*3) + i*(3*3) + 2];
        // and more rows ...

        // bias
        float bias_oi = conv0_bias[o];
    }
}
```

A fully connected layer (fc) is defined as `[out_features, in_features]`. It takes a tensor of shape `[N, in_features]` as input, and outputs a tensor of shape `[N, out_features]`. `N` is denoted as batch size, batch size is 1 if there is one image in the input. The calculation of the fully connected layer is matrix multiplication. 
For the weight matrix of shape `[out_features, in_features]`, you can iterate as follows:
```cpp
for (int o = 0; o < out_features; ++o) {
    for (int i = 0; i < in_features; ++i) {
        float w_oi = fc0_weight[o*out_features + i];
        // ...
    }
    float bias = fc0_bias[o];
}
```

## Example Output

We provide a demo to output scores as an example in [demo.py](./demo.py) using PyTorch (>= 1.6.0) and two sample images in [samples](./samples). You can run the demo and get the confidence scores as follows:
```shell
$ python demo.py --img ./samples/face.jpg
bg score: 0.007086, face score: 0.992914.

$ python demo.py --img ./samples/bg.jpg 
bg score: 0.999996, face score: 0.000004.
```

# Acknowledgement
Thank [Yuantao Feng](https://github.com/fengyuentau) to train the model. 
