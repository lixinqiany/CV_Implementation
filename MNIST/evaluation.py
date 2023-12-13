from model import MNIST_NET
from dataPreparation import MNIST_Dataset
from torch.utils.data import DataLoader
import torch

config = {
    "batch_size": 128,
    "learning_rate":0.01,
    "device": "mps",
    "epoch": 100
}

testingSet = MNIST_Dataset(transform=True, trained=False)
test_loader = DataLoader(testingSet, batch_size=config["batch_size"], shuffle=False)
model = MNIST_NET().to("mps")
model.load_state_dict(torch.load("./output/parameters.pkl"))

############
# EVAL #
############

model.eval()
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
test_total = 0
test_correct = 0
test_loss = 0
for x,y in test_loader:
    x = x.to(config["device"])
    y = y.to(config["device"])
    with torch.no_grad():
        output = model(x)
        loss = criterion(output, y)
        test_loss += loss.item()
        predicted = torch.argmax(output, 1)
        test_correct += (predicted == y).sum().item()
        test_total += y.size(0)
print(f"Test loss: {test_loss / len(test_loader):.2f}, Test acc: {test_correct / test_total*100: .2f}")
