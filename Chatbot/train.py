# Arquivo: bot/train.py

import nltk
import json
import torch
import torch.nn as nn
import numpy as np  # Importação necessária para usar numpy

nltk.download('punkt')

from torch.utils.data import Dataset, DataLoader

from utils import tokenize, stem, bag_of_words
from model import NeuralNet

with open(r"C:\Users\Flavio Jr\Desktop\Chatbot\data\intents.json", 'r', encoding="utf-8") as f:
    intents = json.load(f)

all_words = []
tags = []
x_y = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        x_y.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []
for (pattern_sentence, tag) in x_y:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    Y_train.append(label)

# Modificação para evitar o aviso de conversão de lista para tensor
X_train = np.array(X_train)  # Converta a lista para um único numpy.ndarray
X_train = torch.tensor(X_train, dtype=torch.float32)  # Agora, converte para tensor

Y_train = torch.tensor(Y_train, dtype=torch.long)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hiperparâmetros
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)
learning_rate = 0.001
num_epochs = 1000

# Inicializando dataset e modelo
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
model = NeuralNet(input_size, hidden_size, output_size)

# Otimizador e função de perda
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Treinamento
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print(f"Final loss: {loss.item():.4f}")

# Salvando modelo treinado
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

# Para salvar o modelo
torch.save(model.state_dict(), "data/data.pth")
print("Treinamento completo. Modelo salvo em data/data.pth")
