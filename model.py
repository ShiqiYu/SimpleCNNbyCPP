import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNReLU, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels=self.in_channels,
                              out_channels=self.out_channels,
                              kernel_size=self.kernel_size,
                              stride=self.stride,
                              padding=self.padding,
                              bias=True)
        self.bn = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def combine_conv_bn(self):
        conv_result = nn.Conv2d(self.in_channels,
                                self.out_channels,
                                self.kernel_size,
                                stride=self.stride,
                                padding=self.padding,
                                bias=True)

        scales = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
        conv_result.bias[:] = (self.conv.bias - self.bn.running_mean) * scales + self.bn.bias
        for ch in range(self.out_channels):
            conv_result.weight[ch, :, :, :] = self.conv.weight[ch, :, :, :] * scales[ch]

        return conv_result

class SimpleCLS(nn.Module):
    def __init__(self, input_size=128, num_cls=2, phase='train'):
        super(SimpleCLS, self).__init__()

        self.input_size = input_size
        self.phase = phase.lower()

        self.backbone = nn.Sequential(
            ConvBNReLU(3, 16, 3, 2, 1),    # 128 -> 64
            nn.MaxPool2d(2, 2),            # 64 -> 32
            ConvBNReLU(16, 32, 3, 1),      # 32 -> 30
            nn.MaxPool2d(2, 2),            # 30 -> 15
            ConvBNReLU(32, 32, 3, 2, 1)    # 15 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=32*8*8,
                      out_features=num_cls,
                      bias=True)
        )

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
        elif self.phase == 'train':
            for m in self.backbone.children():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            for m in self.classifier.children():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        else:
            raise NotImplementedError

    def forward(self, x):
        out = self.backbone(x)
        # out = self.classifier(out.view(x.size(0), -1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out if self.phase == 'train' else self.softmax(out)

    def weights2float_string(self, layer, var_name):
        '''
        Convert the weights and bias into float string
        '''
        if isinstance(layer, nn.Conv2d):
            (out_channels, in_channels, width, height) = layer.weight.size()
            lengthstr_w = '{}*{}*{}*{}'.format(out_channels, in_channels, width, height)
        elif isinstance(layer, nn.Linear):
            (out_features, in_features) = layer.weight.size()
            out_channels, in_channels = out_features, in_features
            lengthstr_w = '{}*{}'.format(out_channels, in_channels)
        lengthstr_b = '{}'.format(out_channels)


        w = layer.weight.detach().numpy().reshape(-1)
        b = layer.bias.detach().numpy().reshape(-1)

        resultstr = 'float ' + var_name + '_weight[' + lengthstr_w + '] = {'
        for idx in range(w.size - 1):
            resultstr += (str(w[idx]) + 'f, ')
        resultstr += str(w[-1])
        resultstr += '};\n'

        resultstr += 'float ' + var_name + '_bias[' + lengthstr_b + '] = {'
        for idx in range(b.size - 1):
            resultstr += (str(b[idx]) + 'f, ')
        resultstr += str(b[-1])
        resultstr += '};\n'

        return resultstr, 1

    def port2cpp(self, filename):
        '''Export model weights into a cpp file.
        Modified from https://github.com/ShiqiYu/libfacedetection.train/blob/8155ed20b13c1d432d802b6a1851fb7b2248d568/tasks/task1/yufacedetectnet.py#L196-L255.
        '''
        result_str = '// Auto generated data file\n\n'
        result_str += '''
typedef struct conv_param {
    int pad;
    int stride;
    int kernel_size;
    int in_channels;
    int out_channels;
    float* p_weight;
    float* p_bias;
} conv_param;

typedef struct fc_param {
    int in_features;
    int out_features;
    float* p_weight;
    float* p_bias;
} fc_param;


'''

        # ConvBNReLU types
        conv_bn_relu = [self.backbone[0], self.backbone[2], self.backbone[4]]
        # nn.Conv2D types
        convs = []
        for c in conv_bn_relu:
            convs.append(c.combine_conv_bn())
        # convert to conv weights into float strings
        num_conv = len(convs)
        for idx in range(num_conv):
            rs, _ = self.weights2float_string(convs[idx], 'conv' + str(idx))
            result_str += rs
            result_str += '\n'

        # Linear layers
        linears = [self.classifier[0]]
        for idx, linear in enumerate(linears):
            rs, _ = self.weights2float_string(linear, 'fc' + str(idx))
            result_str += rs
            result_str += '\n'

        # write info
        result_str += 'conv_param conv_params[' + str(len(convs)) + '] = {\n'
        for idx, layer in enumerate(convs):
            result_str += '    {{{padding}, {stride}, {kernel_size}, {in_channels}, {out_channels}, conv{idx}_weight, conv{idx}_bias}}'.format(
                padding=layer.padding[0],
                stride=layer.stride[0],
                kernel_size=layer.kernel_size[0],
                in_channels=layer.in_channels,
                out_channels=layer.out_channels,
                idx=idx
            )
            if (idx < len(convs) - 1):
                result_str += ','
            result_str += '\n'
        result_str += '};\n'
        result_str += 'fc_param fc_params[' + str(len(linears)) + '] = {\n'
        for idx, layer in enumerate(linears):
            result_str += '    {{{in_channels}, {out_channels}, fc{idx}_weight, fc{idx}_bias}}'.format(
                in_channels=layer.in_features,
                out_channels=layer.out_features,
                idx=idx
            )
            if (idx < len(linears) - 1):
                result_str += ','
            result_str += '\n'
        result_str += '};\n'


        # write the content to a file
        #print(result_str)
        with open(filename, 'w') as f:
            f.write(result_str)
            f.close()

        return True

    def port2onnx(self, filename, input_names=['input'], output_names=['conf']):
        dummy_input = torch.randn(1, 3, 128, 128)
        torch.onnx.export(self, dummy_input, filename, input_names=input_names, output_names=output_names)

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    # init and load net
    net = SimpleCLS()
    state_dict = torch.load('./weights/face_binary_cls.pth')
    net.load_state_dict(state_dict)
    net.eval()

    # port to cpp
    net.port2cpp('./weights/face_binary_cls.cpp')
    # port to onnx
    net.port2onnx('./weights/face_binary_cls.onnx')
