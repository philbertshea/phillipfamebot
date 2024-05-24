import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignoreLetters = ['?', '.', '!', ',']

for intent in intents['intents']:
  for pattern in intent['patterns']:
    wordList = nltk.word_tokenize(pattern)
    words.extend(wordList)
    documents.append((wordList, intent['tag'])) # Add a tuple to documents
    if intent['tag'] not in classes:
      classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
# For every word, if the word is not in ignoreLetters list, lemmatize and add the word
words = sorted(set(words))
# Remove duplicates, sort set of words
classes = sorted(set(classes))
# Remove duplicates, sort set of classes

# Save the sorted, duplicate-removed words and classes into their respective pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
outputEmpty = [0] * len(classes)

# print(words)
# print(classes)

# Documents is a dictionary of tuples. In each tuple, the first element is the wordList (tokenised patterns) and the second element is the tag.
for document in documents:
  bag = []
  word_patterns = document[0]
  # Lemmatize, lower case every word in word_patterns
  word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] 
  # print(word_patterns)
  # For each word in all words, if the word is in word_patterns (lemmatized lower-cased current wordList), append 1 to bag, else append 0. Each number in bag corresponds to one word in "words"
  for word in words:
    if word in word_patterns:
      bag.append(1)
    else:
      bag.append(0)
      
  outputRow = list(outputEmpty)
  # Find tag in classes and return its index (in classes) to ind
  ind = classes.index(document[1])
  # Set the ind-th index of outputRow to 1
  outputRow[ind] = 1
  # Append both the bag and outputRow to the array training
  # print(bag+outputRow)
  training.append(bag+outputRow)

# Training is an array of length len(documents) == total # patterns. Each item is an array of length len(words) [from bag] + len(classes) [from outputRow]. 
# For i=0 thru len(words)-1, training[i] = 1 iff words[i] in word_patterns
# For i=len(words) thru len(classes)-1, training[i] = 1 iff classes[i] = tag

# Shuffle Training Data
random.shuffle(training)
training = np.array(training)

#trainX contains all rows, 0th column to len(words)-1th column
trainX = training[:, :len(words)]
#trainY contains all rows, len(words) to last column
trainY = training[:, len(words):]

# Sequential Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu' ))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5')
print('Done')
