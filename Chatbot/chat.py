import json
import torch
from model import NeuralNet
from utils import tokenize, stem, bag_of_words

# Função para carregar os dados (palavras e tags)
def carregar_dados():
    with open("data/intents.json", 'r', encoding="utf-8") as f:
        intents = json.load(f)
    
    all_words = []
    tags = []
    
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            words = tokenize(pattern)
            all_words.extend(words)
    
    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    
    return all_words, tags

# Função para carregar o modelo treinado
def carregar_modelo():
    all_words, tags = carregar_dados()

    # Atualizar as dimensões de input_size e output_size com base nos dados
    input_size = len(all_words)
    output_size = len(tags)
    hidden_size = 8  # Número de neurônios na camada oculta, ajuste conforme necessário

    # Criar o modelo com as dimensões corretas
    model = NeuralNet(input_size, hidden_size, output_size)

    # Carregar o estado do modelo treinado
    model.load_state_dict(torch.load("data/data.pth"))
    model.eval()  # Coloca o modelo em modo de avaliação
    return model, all_words, tags

# Função para prever a tag com base na entrada do usuário
def prever_tag(model, sentence, all_words, tags):
    # Tokenizar e fazer a bag-of-words
    tokenized_sentence = tokenize(sentence)
    bag = bag_of_words(tokenized_sentence, all_words)
    bag = torch.tensor(bag, dtype=torch.float32).unsqueeze(0)  # Adiciona uma dimensão extra para a batch size

    # Passar a entrada pelo modelo
    output = model(bag)
    _, predicted = torch.max(output, dim=1)

    # Obter a tag correspondente
    tag = tags[predicted.item()]
    return tag

# Função de resposta do chatbot
def chatbot():
    model, all_words, tags = carregar_modelo()
    print("Chatbot pronto para conversar! Digite 'sair' para encerrar.")
    
    while True:
        # Entrada do usuário
        sentence = input("Você: ")

        if sentence.lower() == "sair":
            print("Chatbot encerrado.")
            break

        # Prever a tag e escolher a resposta
        tag = prever_tag(model, sentence, all_words, tags)
        
        # Carregar as respostas com base na tag prevista
        with open("data/intents.json", 'r', encoding="utf-8") as f:
            intents = json.load(f)
        
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = intent['responses']
                print("Chatbot: ", response[0])  # Resposta aleatória da lista de respostas

if __name__ == "__main__":
    chatbot()
