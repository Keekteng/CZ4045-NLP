import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pickle

# Function to save and load Python objects
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    

# Function to split sentences into words and convert the column into a list of list of words
def split_text_into_words(df, column_name):
   # Split the text into words
   temp_df = df.copy()
   temp_df[column_name] = temp_df[column_name].str.split()

   # Convert the column into a list
   word_list = temp_df[column_name].tolist()

   # Return the list
   return word_list


# Function to convert list of sentences (list of list of words) into a matrix format
def createMatrix(sentences, word2Idx):

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []

        for word in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = word2Idx['UNKNOWN_TOKEN']
                unknownWordCount += 1

            # Get the label and map to int
            wordIndices.append(wordIdx)

        dataset.append(wordIndices)

    return dataset


# Function to pad each sentence to a fixed length (max sentence length)
def padding(wordList, max_sentence_length):
    for sentence in wordList:
        num_zeros = max_sentence_length-len(sentence)
        sentence.extend([0]*num_zeros)
    return wordList


# Create DataLoader for batched model training
class CustomDataset(Dataset):
   def __init__(self, data, labels):
       self.data = data
       self.labels = labels

   def __len__(self):
       return len(self.data)

   def __getitem__(self, idx):
       return self.data[idx], self.labels[idx]


# Early stop as regularization to prevent overfitting (stops model training when dev set stops improving)
class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_accuracy = -np.inf

    def early_stop(self, accuracy):
        if accuracy > self.max_accuracy:
            self.max_accuracy = accuracy
            self.counter = 0
        elif accuracy < (self.max_accuracy - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# Function to plot graph for accuracy
def plot_accuracy(train_acc, val_acc):
    f = plt.figure(figsize = (12, 6))
    num_epochs = len(train_acc)
    plt.plot(train_acc, label = "Train Acc")
    plt.plot(val_acc, label = "Val Acc")
    plt.title("Change in Accuracy against Epochs")
    plt.legend()
    plt.xticks(np.arange(0, num_epochs+1, step=1))
    plt.ylabel("Accuracy")
    plt.xlabel("Num epochs")
    plt.show()