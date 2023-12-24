import torch
import torchvision
import torchvision.transforms as transforms

# 設置轉換，用於預處理數據
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 下載並加載 MNIST 測試數據集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加載預訓練模型（這裡以簡單的 CNN 為例)
model = torchvision.models.resnet18(pretrained=True)

# 將模型設置為評估模式
model.eval()

# 使用模型進行推理
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        # 這裡可以添加後處理步驟