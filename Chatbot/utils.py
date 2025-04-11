import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem.porter import PorterStemmer

# Criando o stemmer e o tokenizer
stemmer = PorterStemmer()
tokenizer = TreebankWordTokenizer()

def tokenize(sentence):
    """
    Divide a sentença em palavras/tokens usando o TreebankWordTokenizer
    """
    return tokenizer.tokenize(sentence)

def stem(word):
    """
    Aplica stemming à palavra (reduz à raiz)
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Cria um vetor de bag-of-words:
    1 para palavras presentes na sentença, 0 caso contrário
    """
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag
