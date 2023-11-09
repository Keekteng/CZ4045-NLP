import numpy as np
import random
import re
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Function to extract word and NER tags and split them by sentences
def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0],splits[-1].strip('\n')])

    if len(sentence) >0:
        sentences.append(sentence)
        sentence = []
    return sentences


# define casing s.t. NN can use case information to learn patterns
def getCasing(word, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'
    elif re.search(pattern=r'[a-zA-Z]+-[a-zA-Z]+',string=word): # check for hyphenated words
        casing = 'hyphen'


    return caseLookup[casing]


def batchGenerator(dataset,batch_size):
    batch_dataset = []
    for i in range(0, len(dataset), batch_size):
        batch_dataset.append(dataset[i:i + batch_size])

    for batch in batch_dataset:
        batch_words = []
        batch_cases = []
        batch_chars = []
        batch_labels = []
        for sentence in batch:
            words,cases,chars,labels = sentence
            batch_words.append(words)
            batch_cases.append(cases)
            batch_chars.append(chars)
            batch_labels.append(labels)
        yield np.asarray(batch_words),np.asarray(batch_cases),np.asarray(batch_chars),np.asarray(batch_labels)

def predict_on_test(model,test_dataset):
    for sentence in test_dataset:
        words,cases,chars,labels = sentence
        model.predict()

    



# returns matrix with 1 entry = list of 4 elements:
# word indices, case indices, character indices, label indices
def createMatrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx):

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []

        for word, char, label in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = word2Idx['UNKNOWN_TOKEN']
                unknownWordCount += 1
            charIdx = []
            for x in char:
                if x in char2Idx:
                    charIdx.append(char2Idx[x])
                else:
                    charIdx.append(char2Idx['UNKNOWN_TOKEN'])
            # Get the label and map to int
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx))
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])

        dataset.append([wordIndices, caseIndices, charIndices, labelIndices])

    return dataset



# returns data with character information in format
# [['EU', ['E', 'U'], 'B-ORG'], ...]
def addCharInfo(Sentences):
    for i, sentence in enumerate(Sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0], chars, data[1]]
    return Sentences


# 0-pads all words to length 124
def padding(Sentences):
    for i,sentence in enumerate(Sentences):
        Sentences[i][2] = pad_sequences(Sentences[i][2],52,padding='post')
        add_count = 124-len(Sentences[i][2])
        for j in range(add_count):
            zeros_array = np.zeros((1,Sentences[i][2].shape[1]))
            Sentences[i][2] = np.vstack([Sentences[i][2],zeros_array])
        # pad words with 0 index --> PADDING_TOKEN
        Sentences[i][0] = np.pad(Sentences[i][0],(0,124-len(Sentences[i][0])),'constant')
        # pad caseing with 0 index --> PADDING_TOKEN
        Sentences[i][1] = np.pad(Sentences[i][1],(0,124-len(Sentences[i][1])),'constant')
        # padding token for label is 0
        Sentences[i][3] = np.pad(Sentences[i][3],(0,124-len(Sentences[i][3])),'constant')

    return Sentences

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_f1_score = -np.inf

    def early_stop(self, f1_score):
        if f1_score > self.max_f1_score:
            self.max_f1_score = f1_score
            self.counter = 0
        elif f1_score < (self.max_f1_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False