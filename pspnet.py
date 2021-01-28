import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Dropout
from resnet_dilated import ResNet50

# pool with different bin_size
# interpolate back to input size
# concat
class PSPModule(Layer):
    def __init__(self, num_channels, bin_size_list):
        super(PSPModule, self).__init__()
        self.bin_size_list = bin_size_list
        num_filters        = num_channels // len(bin_size_list)
        self.features      = []
        for i in range(len(bin_size_list)):
            self.features.append(
                fluid.dygraph.Sequential(
                    Conv2D(num_channels, num_filters, 1),
                    BatchNorm(num_filters, act='relu')
                )
            )

    def forward(self, inputs):
        out = [inputs]
        for idx, f in enumerate(self.features):
            x = fluid.layers.adaptive_pool2d(inputs, self.bin_size_list[idx])
            x = f(x)
            x = fluid.layers.interpolate(x, inputs.shape[2::], align_corners=True)
            out.append(x)
        out = fluid.layers.concat(out, axis=1)
        return out

class PSPNet(Layer):
    def __init__(self, num_classes=59, backbone='resnet50'):
        super(PSPNet, self).__init__()
        backbone = ResNet50(pretrained = False)
        self.layer1 = backbone.conv
        self.layer2 = backbone.pool2d_max
        self.layer3 = backbone.layer1
        self.layer4 = backbone.layer2
        self.layer5 = backbone.layer3
        self.layer6 = backbone.layer4

        num_channels = 2048
        # stem: res.conv, res.pool2d_max
        self.pspmodule = PSPModule(num_channels,[1, 2, 3, 6])
        num_channels *= 2
        # psp: 2048 -> 2048*2

        # cls: 2048*2 -> 512 -> num_classes
        self.classifier = fluid.dygraph.Sequential(
            Conv2D(num_channels = num_channels, num_filters = 512, filter_size= 3, padding=1),
            BatchNorm(512, act='relu'),
            Dropout(0.1),
            Conv2D(num_channels=512, num_filters=num_classes, filter_size=1, act='softmax')
        )
        # aux: 1024 -> 256 -> num_classes
        num_channels /= 4
        self.classifier_aux = fluid.dygraph.Sequential(
            Conv2D(num_channels = 1024, num_filters = 256, filter_size= 3, padding=1),
            BatchNorm(256, act='relu'),
            Dropout(0.1),
            Conv2D(num_channels=256, num_filters=num_classes, filter_size=1, act='softmax')
        )
        
    def forward(self, inputs):

        # aux: tmp_x = layer3
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        aux = x
        x = self.layer6(x)
        x = self.pspmodule(x)
        x = self.classifier(x)
        x = fluid.layers.interpolate(x, inputs.shape[2::], align_corners=True)
        aux = self.classifier_aux(aux)
        aux = fluid.layers.interpolate(aux, inputs.shape[2::], align_corners=True)
        return x, aux



def main():
    with fluid.dygraph.guard(fluid.CPUPlace()):
        x_data=np.random.rand(2,3, 473, 473).astype(np.float32)
        x = to_variable(x_data)
        model = PSPNet(num_classes=59)
        model.train()
        pred, aux = model(x)
        print(pred.shape, aux.shape)

if __name__ =="__main__":
    main()
