from dataPreparation import MNIST_Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import MNIST_NET
import torch, time
from util import systemCheck
from tqdm import tqdm

config = {
    "batch_size": 128,
    "learning_rate":0.01,
    "device": systemCheck(),
    "epoch": 100
}

trainingSet = MNIST_Dataset(transform=True)
testingSet = MNIST_Dataset(transform=True, trained=False)
train_loader = DataLoader(trainingSet, batch_size=config["batch_size"], shuffle=True)
test_loader = DataLoader(testingSet, batch_size=config["batch_size"], shuffle=False)

############
# showcase #
############

fig = plt.figure()
for i in range(12):
    plt.subplot(3, 4, i+1)
    plt.tight_layout()
    plt.imshow(trainingSet.data['train'][i], cmap='gray', interpolation='none')
    plt.title("Labels: {}".format(trainingSet.data['train_label'][i]))
    plt.xticks([])
    plt.yticks([])
plt.savefig("showcase.png")
plt.show()

############
# showcase #
############

model = MNIST_NET().to(config["device"])
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.5)  # lr学习率，momentum冲量

for i in range(config["epoch"]):
    model.train()
    train_total = 0
    train_correct = 0
    train_loss = 0

    epoch = tqdm(train_loader)
    for x, y in epoch:
        optimizer.zero_grad()
        x, y = x.to(config["device"]), y.to(config["device"])
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = torch.argmax(output, 1)
        train_correct += (predicted == y).sum().item()
        train_total += y.size(0)
        epoch.set_description(f'Epoch [{i + 1}/{config["epoch"]}]')
        batch_loss = loss.item()
        batch_acc = (predicted == y).sum().item() / y.size(0)
        # epoch.set_postfix(loss=round(batch_loss, 2), acc=round(batch_acc * 100, 2))

    train_loss = train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    print(f"Epoch Loss: {train_loss:.2f}, Epoch acc: {train_accuracy * 100:.2f}%")
    time.sleep(0.5)

############
# EVAL #
############

model.eval()
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

torch.save(model.state_dict(), "output/parameters.pkl")