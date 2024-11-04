import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# 检查CUDA/ROCm是否可用，如果不可用则使用CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.version.hip is not None:
    device = torch.device('rocm')
else:
    print("Neither CUDA nor ROCm is available. Defaulting to CPU.")
    device = torch.device('cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

import torch.nn as nn
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(30):  # 训练 5 个 epoch
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

torch.save(model.state_dict(), 'mnist_cnn.pth')

print(f'Accuracy: {100 * correct / total}%')
#
# # 加载模型
# model = CNN()
# model.load_state_dict(torch.load('mnist_cnn.pth'))
# model.eval()  # 设置为评估模式
#
# from PIL import Image
# import torchvision.transforms as transforms
#
#
# def predict_digit(image_path):
#     # 读取图像并转换为灰度图
#     image = Image.open(image_path).convert('L')
#
#     # 数据预处理：缩放到 28x28，并进行归一化
#     transform = transforms.Compose([
#         transforms.Resize((28, 28)),  # 调整大小
#         transforms.ToTensor(),  # 转换为 Tensor
#         transforms.Normalize((0.5,), (0.5,))  # 归一化
#     ])
#
#     image = transform(image)
#     image = image.unsqueeze(0)  # 添加一个批次维度
#
#     # 使用模型进行预测
#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output.data, 1)
#
#     return predicted.item()
#
#
# # 测试手写数字
# image_path = 'path_to_your_digit_image.png'  # 替换为你的图像路径
# predicted_digit = predict_digit(image_path)
# print(f'Predicted Digit: {predicted_digit}')
