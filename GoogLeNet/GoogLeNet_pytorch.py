# importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x, inplace=True)

class Inception(nn.Module):
	def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, 
		conv_block=None):
		super(Inception, self).__init__()
		if conv_block is None:
			conv_block = BasicConv2d

		self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
		self.branch2 = nn.Sequential(conv_block(in_channels, ch3x3red, kernel_size=1),
			conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1))
		self.branch3 = nn.Sequential(conv_block(in_channels, ch5x5red, kernel_size=1),
			conv_block(ch5x5red, ch5x5, kernel_size=5, padding=1))
		self.branch4 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
			conv_block(in_channels, pool_proj, kernel_size=1))

	def forward(self, x):
		branch1 = self.branch1(x)
		branch2 = self.branch2(x)
		branch3 = self.branch3(x)
		branch4 = self.branch4(x)

		outputs = [branch1, branch2, branch3, branch4]
		return torch.cat(outputs, 1)

class GoogLeNet(nn.Module):
	def __init__(self, width = 224, height = 224, depth = 3, num_classes = 1000):
		super(GoogLeNet, self).__init__()
		blocks = [BasicConv2d, Inception]

		conv_block = blocks[0]
		inception_block = blocks[1]

		self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
		self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
		self.conv2 = conv_block(64, 64, kernel_size=1)
		self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
		self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

		self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
		self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
		self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

		self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
		self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
		self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
		self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
		self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
		self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

		self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
		self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.dropout = nn.Dropout(0.4)
		self.fc = nn.Linear(1024, num_classes)

	def forward(self, x):
		# type: (Tensor) -> GoogLeNetOutputs

		x = self.conv1(x) # N x 3 x 224 x 224
		x = self.maxpool1(x) # N x 64 x 112 x 112
		x = self.conv2(x) # N x 64 x 56 x 56
		x = self.conv3(x) # N x 64 x 56 x 56
		x = self.maxpool2(x) # N x 192 x 56 x 56
		x = self.inception3a(x) # N x 192 x 28 x 28
		x = self.inception3b(x) # N x 256 x 28 x 28
		x = self.maxpool3(x) # N x 480 x 28 x 28
		x = self.inception4a(x) # N x 480 x 14 x 14
		x = self.inception4b(x) # N x 512 x 14 x 14
		x = self.inception4c(x) # N x 512 x 14 x 14
		x = self.inception4d(x) # N x 512 x 14 x 14
		x = self.inception4e(x) # N x 528 x 14 x 14
		x = self.maxpool4(x) # N x 832 x 14 x 14
		x = self.inception5a(x) # N x 832 x 7 x 7
		x = self.inception5b(x) # N x 832 x 7 x 7
		x = self.avgpool(x) # N x 1024 x 7 x 7
		x = torch.flatten(x, 1) # N x 1024 x 1 x 1
		x = self.dropout(x) # N x 1024
		x = self.fc(x) # N x 1000 (num_classes)
		return x

net = GoogLeNet(width = 224, height = 224, depth = 3, num_classes = 1000)
print(net)