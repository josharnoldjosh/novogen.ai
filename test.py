#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:19:48 2018

@author: josharnold
"""

# Load data
print("Loading data...")
import pandas as pd
data = pd.read_csv('data.txt', sep=" ", header=None) # Load almost two million fucking molecules! 
data = data.iloc[1:] # Drop first row which is a header
data = data.iloc[0:50000] # Limit molecules
print("Done!\n")

# Parse smiles from data
print("Creating smiles...")   
smiles = []
minLen = 10000
maxLen = 0
for d in data[0]:
	split_data = d.split()
	molecule = split_data[1] + "\n" # Add end of line character, \n
	smiles.append(molecule)
	if len(molecule) > maxLen:
		maxLen = len(molecule)
	elif len(molecule) < minLen:
		minLen = len(molecule)
print("Done!\n")   

# Create list of possible chars
print("Learning vocab...")  
vocab = []
for smile in smiles:
    for char in smile:
        if char not in vocab:
            vocab.append(char)  
print("Done!\n")                  

# Encode vocab
print("Encoding vocab...")
char_to_int = dict((c, i) for i, c in enumerate(vocab))
int_to_char = dict((i, c) for i, c in enumerate(vocab))
print("Done!\n")

# Generate patterns
print("Generating patterns...")  
X_data = []
y_data = []
seq_len = 20
step = 1
for smile in smiles:
    for i in range(0, len(smile) - seq_len, step):
        seq_in = smile[i: i + seq_len]
        seq_out = smile[i + seq_len]
        X_data.append([char_to_int[char] for char in seq_in])
        y_data.append(char_to_int[seq_out])
print("Done! Total patterns:", len(X_data), "\n")  

# One hot encoding
print("Begining one hot encoding...") 
import numpy
from keras.utils import np_utils
# reshape X to be [samples, time steps, features]
X = numpy.reshape(X_data, (len(X_data), seq_len, step))
# normalize
X = X / float(len(vocab))
# one hot encode the output variable
y = np_utils.to_categorical(y_data)
print("Done!\n")

# Define the LSTM model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(1024))
model.add(Dropout(0.2))

model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Define the checkpoint
filepath="checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)

# load the network weights
#filename = "weights-improvement-01-1.1467.hdf5"
#model.load_weights("checkpoints/"+filename)
#model.compile(loss='categorical_crossentropy', optimizer='adam')

# Pick a random seed for generating molecules
import sys
start = numpy.random.randint(0, len(X_data)-1)
pattern = X_data[start]
print("Seed:")
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

# Generate molecules
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(len(vocab))
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print("\nDone.")
