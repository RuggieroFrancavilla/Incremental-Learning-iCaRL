import torch
import numpy as np
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

"""
Credits to @hshustc
Taken from https://github.com/hshustc/CVPR19_Incremental_Learning/tree/master/cifar100-class-incremental
"""


class PatchShuffle(nn.Module):
    def __init__(self, patch_size=(2,2), shuffle_probability=0.05):
        super().__init__()
        self.patch_size = patch_size
        self.shuffle_probability = shuffle_probability

    def forward(self, T: torch.Tensor) -> torch.Tensor:
        if T.shape[0] != 128:
            # case in which we are evaluating the model (PatchShuffle is NOT applied)
            return T
        patch_size = self.patch_size
        shuffle_probability = self.shuffle_probability
        input_n, input_c, input_h, input_w = T.shape

        T = T.reshape(-1, input_h, input_w)

        # for each feature maps, decide whether to patchshuffle or not
        indices_tensor = []    # will be converted into a (input_n*input_c, input_h, input_w) tensor
        for i in range(T.shape[0]):
            # create the indices map
            idx = np.arange(input_h*input_w).reshape(input_h,input_w)

            flick_patchshuffle = np.random.choice( [True,False], p=(self.shuffle_probability, 1-self.shuffle_probability) )
            if flick_patchshuffle:
                w_patches = input_w // patch_size[1]    # n. patches along width
                h_patches = input_h // patch_size[0]    # n. patches along height
                patches_idx = idx.reshape(h_patches,patch_size[0],w_patches,patch_size[1]).swapaxes(1,2).reshape(-1,patch_size[0],patch_size[1])
                for i,patch_idx in enumerate(patches_idx):
                    patches_idx[i] = np.random.permutation(patch_idx.reshape(-1)).reshape(2,2)


                final = []

                for i in range(h_patches):
                    block_row = []
                    for j in range(w_patches):
                        block_row.append(patches_idx[w_patches*i+j])
                    block_row = np.hstack(block_row)
                    final.append(block_row)
                idx = np.vstack(final)
            
            indices_tensor.append(idx)

        indices_tensor = np.array(indices_tensor).reshape(input_n,input_c,input_h,input_w)

        return T.reshape(-1)[indices_tensor.reshape(-1)].reshape(input_n, input_c, input_h, input_w)
    
    def extra_repr(self) -> str:
        return ('patch_size={patch_size}, shuffle_probability={shuffle_probability}')







def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.patchshuffle1 = PatchShuffle()
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.patchshuffle2 = PatchShuffle()
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.patchshuffle1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.patchshuffle2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class InitialBlock(nn.Module):
    def __init__(self):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.patchshuffle = PatchShuffle()
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.patchshuffle(out)
        out = self.bn(out)
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, initial_block, block, layers, num_classes=10):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.feature_extractor = nn.Sequential(self._make_initial_layer(initial_block),
          self._make_layer(block, 16, layers[0]),
          self._make_layer(block, 32, layers[1], stride=2),
          self._make_layer(block, 64, layers[2], stride=2),
          nn.AvgPool2d(8, stride=1))

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                PatchShuffle(),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def _make_initial_layer(self, block):
        layers = []
        layers.append(block())

        return nn.Sequential(*layers)
        

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def get_features(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)

        return x


def resnet32(pretrained=False, **kwargs):
    n = 5
    model = ResNet(InitialBlock, BasicBlock, [n, n, n], **kwargs)
    return model
