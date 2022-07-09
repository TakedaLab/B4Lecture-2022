import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


class Sound_Data(Dataset):
    def __init__(self, meta_data, phase, maxLength=None) -> None:
        self.len = meta_data.shape[0]
        self.data = []
        self.labels = []

        for i in tqdm(range(self.len)):
            wave, _ = librosa.load(meta_data.loc[i, "path"], sr=44100)

            self.data.append(wave)
            self.labels.append(meta_data.loc[i, "label"])
        sr = 44100
        # 最長音声の長さ
        if phase == "train":
            self.maxLength = np.max([len(self.data[i]) for i in range(self.len)])
        else:
            self.maxLength = maxLength

        # zero padding & mel_spectrogram
        for i in tqdm(range(self.len)):
            self.data[i] = np.pad(self.data[i], (0, self.maxLength - len(self.data[i])))
            mel_spec = librosa.feature.melspectrogram(y=self.data[i], sr=sr)
            self.data[i] = librosa.power_to_db(mel_spec, ref=np.max)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.data[index], self.labels[index]


class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(50176, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def main():
    meta_train_file = "training.csv"
    meta_test_file = "test_truth.csv"

    meta_train_data = pd.read_csv(meta_train_file)
    meta_test_data = pd.read_csv(meta_test_file)

    if os.path.exists("melspec_train_loader") and os.path.exists("melspec_test_loader"):
        train_loader = pickle.load(open("melspec_train_loader", "rb"))
        test_loader = pickle.load(open("melspec_test_loader", "rb"))
    else:
        train_data = Sound_Data(meta_train_data, "train")
        maxLength = train_data.maxLength
        test_data = Sound_Data(meta_test_data, "test", maxLength)

        train_loader = DataLoader(train_data, batch_size=256)
        test_loader = DataLoader(test_data, batch_size=256)

        with open("melspec_train_loader", "wb") as w:
            pickle.dump(train_loader, w)
        with open("melspec_test_loader", "wb") as w:
            pickle.dump(test_loader, w)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = CNN().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        running_loss = 0
        correct = 0
        for (inputs, labels) in dataloader:
            inputs, labels = inputs[:, np.newaxis, :, :].to(device), labels.to(device)

            pred = model(inputs)
            loss = loss_fn(pred, labels)
            running_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        train_loss = running_loss / size
        correct = correct / size
        print(
            f"Training: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n"
        )
        return correct

    def test(dataloader, model):
        size = len(dataloader.dataset)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = (
                    inputs[:, np.newaxis, :, :].to(device),
                    labels.to(device),
                )
                pred = model(inputs)
                test_loss += loss_fn(pred, labels).item()
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(
            f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        return correct

    epochs = 100
    train_corrects = np.zeros(epochs)
    test_corrects = np.zeros(epochs)
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------")
        train_corrects[t] = train(train_loader, model, loss_fn, optimizer)
        test_corrects[t] = test(test_loader, model)
    print("Done!")

    fig = plt.figure()
    plt.plot(np.arange(epochs), train_corrects, c="blue", label="train")
    plt.plot(np.arange(epochs), test_corrects, c="orange", label="test")
    plt.title(
        f"result\nAccuracy(Training): {(100*train_corrects[t]):>0.1f}%\nAccuracy(Test): {(100*test_corrects[t]):>0.1f}%"
    )
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("result.png")
    plt.show()


if __name__ == "__main__":
    main()

