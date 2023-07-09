import nltk
from nltk.stem import PorterStemmer
import json
import pickle
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')

words = []
classes = []
pattern_word_tags_list = []
ignore_words = ['?', '!', ',', '.', "'s", "'m"]

train_data_file = open('intents.json')
data = json.load(train_data_file)
train_data_file.close()

stemmer = PorterStemmer()

def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            stemmed_word = stemmer.stem(word.lower())
            stem_words.append(stemmed_word)
    return stem_words

def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):
    for intent in data['intents']:
        for pattern in intent['patterns']:
            pattern_words = nltk.word_tokenize(pattern)
            words.extend(pattern_words)
            pattern_word_tags_list.append((pattern_words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
    stem_words = get_stem_words(words, ignore_words)
    stem_words = list(set(stem_words))
    stem_words.sort()
    classes.sort()
    print('stem_words list:', stem_words)
    return stem_words, classes, pattern_word_tags_list

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []
    for word_tags in pattern_word_tags_list:
        pattern_words = word_tags[0]
        bag_of_words = []
        stemmed_pattern_words = get_stem_words(pattern_words, ignore_words)
        for word in stem_words:
            if word in stemmed_pattern_words:
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        bag.append(bag_of_words)
    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    labels = []
    for word_tags in pattern_word_tags_list:
        labels_encoding = [0] * len(classes)
        tag = word_tags[1]
        tag_index = classes.index(tag)
        labels_encoding[tag_index] = 1
        labels.append(labels_encoding)
    return np.array(labels)

def preprocess_train_data():
    stem_words, tag_classes, word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)
    with open('stem_words.pkl', 'wb') as f:
        pickle.dump(stem_words, f)
    with open('tag_classes.pkl', 'wb') as f:
        pickle.dump(tag_classes, f)
    train_x = bag_of_words_encoding(stem_words, word_tags_list)
    train_y = class_label_encoding(tag_classes, word_tags_list)
    return train_x, train_y

bow_data, label_data = preprocess_train_data()

print("first BOW encoding:", bow_data[0])
print("first Label encoding:", label_data[0])