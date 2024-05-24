import random
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('mybot/botfiles/intents.json').read())

words = pickle.load(open('mybot/botfiles/words.pkl', 'rb'))
classes = pickle.load(open('mybot/botfiles/classes.pkl', 'rb'))
model = tf.keras.models.load_model('mybot/botfiles/chatbot_model.h5')


# clean_up_sentence tokenizes sentence (breaks into words), lemmatizes it (find root word for each word), then returns this
def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
  return sentence_words


def bag_of_words(sentence):
  sentence_words = clean_up_sentence(sentence)
  # Initialise bag to len(words) number of zeros
  bag = [0] * len(words)
  # For each word in sentence_words, if the word is in words, set bag[i] = 1
  for w in sentence_words:
    for i, word in enumerate(words):
      if word == w:
        bag[i] = 1
  return np.array(bag)


def predict_class(sentence):
  bagWords = bag_of_words(sentence)
  # Get prediction of which class bagWords belongs to (result is an array of probabilities: result[i] = probability of being in class i)
  result = model.predict(np.array([bagWords]))[0]
  print('result')
  print(result)
  ERR = 0.25
  # Only take in class if probability > ERR
  results = [[i, r] for i, r in enumerate(result) if r > ERR]

  # Sort by the r value x[1] (enum in [i, r]), descending order
  results.sort(key=lambda x: x[1], reverse=True)

  return_list = []
  # For each result r, store the intent = class i and probability = r as a string
  for r in results:
    return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
  return return_list


def get_response(intents_list, intents_json):
  # Get the tag of the intent with the highest probability
  tag = intents_list[0]['intent']
  list_of_intents = intents_json['intents']
  # For each intent in intents_json, if tag matches the highest probability intent tag, return result
  for i in list_of_intents:
    if i['tag'] == tag:
      result = random.choice(i['responses'])
      break
  return result


def query(message):
  # Run predict_class
  ints = predict_class(message)
  # Get response using ints - intents_list generated from predict_class and intents json
  res = get_response(ints, intents)
  print(res)
  return res
