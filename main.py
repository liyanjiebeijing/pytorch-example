import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os

# 数据集参数
DATA_ROOT = '/root/data/cifar100/'
BATCH_SIZE = 128
NUM_WORKERS = 4

# 训练参数
EPOCHS = 100
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# 定义ResNet18修改版（适配CIFAR-100的32x32输入）
class CIFAR100ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        # 修改第一个卷积层和最大池化层
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # 移除原始的最大池化层
        # 修改最后的全连接层
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 数据预处理
def get_datasets():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                            (0.2675, 0.2565, 0.2761)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                            (0.2675, 0.2565, 0.2761)),
    ])

    # 自动下载数据集
    train_set = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=False, download=True, transform=test_transform)

    return train_set, test_set

def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter("logs/cifar100_resnet18")

    # 获取数据集
    train_set, test_set = get_datasets()

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    # 初始化模型
    model = CIFAR100ResNet18().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR,
                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[60, 120], gamma=0.1)

    best_acc = 0.0

    # 训练循环
    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": optimizer.param_groups[0]['lr']
            })

        # 验证阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # 计算指标
        train_loss = train_loss / len(train_set)
        test_loss = test_loss / len(test_set)
        test_acc = 100. * correct / total

        # 学习率更新
        scheduler.step()

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_resnet18_cifar100.pth")

        # 记录TensorBoard指标
        writer.add_scalars("Loss", {
            "train": train_loss,
            "test": test_loss
        }, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        # 打印进度
        tqdm.write(f"Epoch {epoch+1:03d}: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Accuracy: {test_acc:.2f}% | "
                  f"Best Acc: {best_acc:.2f}%")

    writer.close()
    print(f"Training complete! Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
