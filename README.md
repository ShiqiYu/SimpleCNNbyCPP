# SimpleCNNbyCPP
For Course CS205 'C/C++ Program Design' at Southern University of Scicence and Technology, China.

Model Information
- The model is trained to perform face classification.
- The model takes an 128x128 RGB image as input, and outputs two confidences for background and face respectively.

The model is defined in [model.py](./model.py). Visualization of the model: [netron](https://netron.app/?url=https://raw.githubusercontent.com/ShiqiYu/SimpleCNNbyCPP/main/weights/face_binary_cls.onnx).

***NOTE***: The parameters of batch normalization is already combined to convolutional layers' when porting weights (`.pth`) to `.cpp`.

<!-- https://netron.app/?url=https://raw.githubusercontent.com/chandrikadeb7/Face-Mask-Detection/master/face_detector/deploy.prototxt -->

# Acknowledgement
Thank [Yuantao Feng](https://github.com/fengyuentau) to train the model. 
