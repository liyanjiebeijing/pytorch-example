import argparse
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

DATA_ROOT = '/root/data/cifar100/'

FP_16=True

class CIFAR100ResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def get_datasets():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    train_set = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=train_transform)
    test_set = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=test_transform)
    return train_set, test_set

def train(rank, world_size, args):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    # 初始化模型
    model = CIFAR100ResNet18().to(rank)
    model = DDP(model, device_ids=[rank])

    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # 恢复训练
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.resume, map_location=map_location)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

    # 数据加载
    train_set, test_set = get_datasets()
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # 只在主进程初始化TensorBoard
    if rank == 0:
        writer = SummaryWriter("logs/cifar100_resnet18")

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        # 训练阶段
        train_loss = 0.0
        if rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(rank), labels.to(rank)

            if FP_16:
                # 混合精度前向
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # 梯度缩放反向传播
                scaler.scale(loss).backward()

                # 参数更新（自动同步梯度）
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            if rank == 0:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}",
                                "lr": optimizer.param_groups[0]['lr']})

        # 验证阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(rank), labels.to(rank)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # 同步所有进程的指标
        torch.distributed.barrier()
        total = torch.tensor(total, device=rank)
        correct = torch.tensor(correct, device=rank)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)

        test_acc = 100. * correct / total
        test_loss = test_loss / len(test_loader)
        train_loss = train_loss / len(train_loader)

        scheduler.step()

        # 主进程记录日志和保存模型
        if rank == 0:
            writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'best_acc': best_acc
                }, "best_model.pth")

            pbar.close()
            print(f"Epoch {epoch+1:03d} | "
                 f"Train Loss: {train_loss:.4f} | "
                 f"Test Loss: {test_loss:.4f} | "
                 f"Accuracy: {test_acc:.2f}% | "
                 f"Best Acc: {best_acc:.2f}%")

    if rank == 0:
        writer.close()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    world_size = args.gpus * args.nodes
    mp.spawn(train, args=(world_size, args), nprocs=args.gpus)
