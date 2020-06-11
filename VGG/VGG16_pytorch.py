# importing libraries
import torch
import torch.nn as nn

class VGG16(nn.Module):
	def __init__(self, num_classes = 1000):
		super(VGG16, self).__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2),

			nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
			nn.ReLU(inplace = True),
			nn.MaxPool2d(kernel_size = 2, stride = 2))
		
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLU(inplace = True),
			nn.Dropout(p=0.5),

			nn.Linear(4096, 4096),
			nn.ReLU(inplace = True),
			nn.Dropout(p=0.5),

			nn.Linear(4096, num_classes))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), 512 * 7 * 7)
		x = self.classifier(x)
		return x

def vgg(**kwargs):
	model = VGG16(**kwargs)
	return model

print(vgg(num_classes = 1000))