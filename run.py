import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


# 定义 CNN 模型
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


# 加载训练好的模型
model = CNN()
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()


# 画板应用程序
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字画板")
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(root, text="预测数字", command=self.predict_digit)
        self.button_predict.pack()

        self.button_clear = tk.Button(root, text="清除画板", command=self.clear_canvas)
        self.button_clear.pack()

        self.canvas.bind("<B1-Motion>", self.paint)

        self.image = Image.new("L", (200, 200), color="white")  # 创建白色背景
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x1, y1 = (event.x - 5), (event.y - 5)
        x2, y2 = (event.x + 5), (event.y + 5)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def predict_digit(self):
        # 保存绘图并进行预测
        self.image = self.image.resize((28, 28)).convert('L')  # 调整为 28x28 大小
        self.image = Image.eval(self.image, lambda x: 255 - x)  # 反色，使数字为白色背景为黑色

        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image_tensor = transform(self.image).unsqueeze(0)  # 添加批次维度

        # 预测
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)
            print(f'预测数字: {predicted.item()}')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), color="white")
        self.draw = ImageDraw.Draw(self.image)


# 启动应用
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
