# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from sklearn.feature_extraction import stop_words
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

# restore all of our data structures
import pickle
data = pickle.load( open( "/home/saraswathi/work/PycharmProjects/bank1/training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('/home/saraswathi/Desktop/testt.json') as json_data:
    intents = json.load(json_data)

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='/home/saraswathi/work/PycharmProjects/bank1/tflearn_logs')
def clean_up_sentence(sentence):
    # tokenize the pattern
    words = nltk.word_tokenize(sentence)
    #stopwords = ('what','why','ing','when','is','can','are','have','has','you','your','how','should','would','it','they','them','those','the','a','an','was','where','could','himself')
    stop = (stop_words.ENGLISH_STOP_WORDS)
    sentence_words = [word for word in words if word not in stop]
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    print(sentence_words)
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

p = bow("hi?", words)

# load our saved model
model.load('/home/saraswathi/work/PycharmProjects/bank1/model/model.tflearn')

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=True):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if ['tag'] == results[0][0]:
                        if show_details: print ('tag', i['tag'])
                        # a random response from the intent
                        print(random.choice(i['responses']))

            results.pop(0)


response('savings account means?')


