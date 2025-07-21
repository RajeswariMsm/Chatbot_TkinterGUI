# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 17:50:40 2025

@author: DELL
"""

# Import the Natural Language Toolkit for text processing

# Import WordNetLemmatizer to reduce words to their base form
from nltk.stem import WordNetLemmatizer
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
# Import numpy for numerical operations
import numpy as np
# Import Sequential model from Keras for building neural networks
from tensorflow.keras.models import Sequential
# Import layers for neural network architecture
from tensorflow.keras.layers import Dense, Activation, Dropout
# Import Stochastic Gradient Descent optimizer
from tensorflow.keras.optimizers import SGD
# Import random for generating random numbers
import random
# Import json for handling JSON data
import json
# Import pickle for serializing and deserializing Python objects
import pickle
import nltk
# Initialize empty lists to store processed data
words=[]  # Will store all unique words from patterns
classes = []  # Will store all unique quest tags
documents = []  # Will store (pattern, quest) pairs
ignore_words = ['?', '!']  # Punctuation marks to be ignored during processing

# Load the quests data from JSON file
data_file = open('quest.json').read() # Open and read the JSON file containing quest data
quests = json.loads(data_file)  # Parse JSON into Python dictionary
# Loop through each intent in the intents data structure
for quest in quests['quests']:
    # Loop through each pattern phrase in the current quest
    for pattern in quest['patterns']:

        # Tokenize each pattern into individual words
        w = nltk.word_tokenize(pattern)
        # Add these tokenized words to our global words list
        words.extend(w)
        # Create a document entry pairing the tokenized pattern with its quest tag
        documents.append((w, quest['tag']))

        # Track unique quest tags by adding new ones to the classes list
        if quest['tag'] not in classes:
            classes.append(quest['tag'])

# Process all collected words: lemmatize them, convert to lowercase, and remove ignored words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# Remove duplicate words, convert to a sorted list for consistent ordering
words = sorted(list(set(words)))
# Ensure classes (intent tags) are unique and sorted alphabetically
classes = sorted(list(set(classes)))
# Print summary statistics about the processed training data
# documents = combination between patterns and quests
print (len(documents), "documents")
# classes = quests
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)
# Save processed words and classes to pickle files for later use
pickle.dump(words,open('words1.pkl','wb'))
pickle.dump(classes,open('classes1.pkl','wb'))

# Create our training data
training = []
# Create an empty array for our output with zeros (one position for each class)
output_empty = [0] * len(classes)
# Process each document (pattern-tag pair) in our documents list
for doc in documents:
    # Initialize our bag of words vector (features)
    bag = []
    # Get the tokenized words for the current pattern
    pattern_words = doc[0]
    # Lemmatize each word - convert to base form to handle different word forms
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # Create the bag-of-words representation: 1 if word exists in pattern, 0 otherwise
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # Prepare output vector: all zeros except 1 for the current tag's position
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    # Add the feature vector and its corresponding output to training data
    training.append([bag, output_row])
    # shuffle our features and turn into np.array
random.shuffle(training)  # randomize the order of training data to prevent learning sequence patterns
training = np.array(training, dtype=object)  # convert list to numpy array for easier manipulation

# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])  # extract features (tokenized words) from first column
train_y = list(training[:,1])  # extract labels (one-hot encoded intents) from second column
print("Training data created")
# Create model - 3 layers. First layer 128 neurons, 
# second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
# Input layer with 128 neurons, ReLU activation, shape based on training data features
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Add dropout of 50% to prevent overfitting
model.add(Dropout(0.5))
# Hidden layer with 64 neurons and ReLU activation
model.add(Dense(64, activation='relu'))
# Another dropout layer to further reduce overfitting
model.add(Dropout(0.5))
# Output layer with neurons matching the number of intent classes
# Softmax activation for multi-class classification (converts to probabilities)
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov 
# accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Configure the model with categorical crossentropy loss function (suitable for multi-class classification)
# and accuracy as the evaluation metric
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model on our training data
# epochs=200: The model will see the entire dataset 200 times
# batch_size=5: Update weights after seeing 5 samples
# verbose=1: Show progress during training
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# Save the trained model to disk for later use
# Also save the training history in the model file
model.save('chatbot_model.h5', hist)

print("model created and saved")