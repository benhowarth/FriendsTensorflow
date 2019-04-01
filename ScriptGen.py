from ScriptGet import *
import random


characterToTrain = "joey"

characterToTrainId = getCharacterId(characterToTrain)
lines_to_learn = list(filter(lambda x: (characterToTrainId in x.charIds) and (x.type == 3), lines))

just_text = list(map(lambda x: x.text, lines_to_learn))

just_text_string = "".join(just_text)

word_ends = [".", "?", "!", ":", ";", "(", ")", " ", "[", "]", "'", "/", ","]
wordList = []


def splitPunc(word):
    if (len(word) > 1):
        for end in word_ends:
            # punctuation index
            pi = word.find(end)
            if (pi > -1):
                if (len(word[:pi]) == 0):
                    return [word[pi]] + splitPunc(word[pi + 1:])
                elif (len(word[pi + 1:]) == 0):
                    return splitPunc(word[:pi]) + [word[pi]]
                else:
                    return splitPunc(word[:pi]) + [word[pi]] + splitPunc(word[pi + 1:])

    return [word]


wordList = splitPunc(just_text_string)

chars = sorted(list(set(wordList)))
# chars=sorted(list(set(just_text_string)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 30
step = 2
sentences = []
next_chars = []

for i in range(0, len(wordList) - maxlen, step):
    sentences.append(wordList[i:i + maxlen])
    next_chars.append(wordList[i + maxlen])

import tensorflow as tf
import numpy as np


x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
import sys
import io
from time import time
from keras.callbacks import TensorBoard

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

lr = 0.005
optimizer = RMSprop(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

epochsToTrain = 50


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


diversities = [0.2, 0.5, 1.0, 1.2]
sentenceLenToGen = 60


def on_epoch_end(epoch, logs):
    if epoch + 1 == 1 or epoch + 1 == epochsToTrain:
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        start_index = random.randint(0, len(wordList) - maxlen - 1)
        stringToFile = "EPOCH " + str(epoch) + "\n\n"
        for diversity in diversities:
            print('diversity:', diversity)
            generated = ''
            sentence = wordList[start_index:start_index + maxlen]
            generated += "".join(sentence)
            sys.stdout.write(generated)
            for i in range(sentenceLenToGen):
                x_pred = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:]
                sentence.append(next_char)

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
            stringToFile += "DIVERSITY: " + str(diversity) + "\n" + generated + "\n\n\n"
        f = open("{}/output.txt".format(folderName), "a")
        f.write(stringToFile)
        f.close()
    else:
        print()
        print('----- Not generating text after Epoch: %d' % epoch)


generate_text = LambdaCallback(on_epoch_end=on_epoch_end)

params = {}
params["lr"] = lr
params["diversities"] = ",".join([str(d) for d in diversities])
params["sentenceLenToGen"] = sentenceLenToGen
params["maxlen"] = maxlen
params["step"] = step
params["epochsToTrain"] = epochsToTrain
params["characterToTrain"] = characterToTrain
params["characterToTrainId"] = characterToTrainId
params["epCount"] = epCount


maxCellWidth = 30
maxTableWidth = ((maxCellWidth * 2) + 3)


def dict_to_table(dictionary, title=""):
    res = "=" * maxTableWidth
    res += "\n"
    res += "| " + title + (" " * (maxTableWidth - 1 - len(title))) + "|"
    res += "\n"
    res += "=" * maxTableWidth
    res += "\n"
    for k, v in dictionary.items():
        kStr = str(k)
        vStr = str(v)
        res += "|" + kStr + (" " * (maxCellWidth - len(kStr))) + "|" + vStr + (" " * (maxCellWidth - len(vStr))) + "|"
        res += "\n"

    res += "=" * maxTableWidth
    res += "\n"
    return res


class TBLog(TensorBoard):
    def __init__(self, log_dir):
        super().__init__(log_dir=log_dir)

    def on_train_end(self, logs=None):
        f = open("{}/params.txt".format(folderName), "a")
        f.write(dict_to_table(params, "Hyperparameters & additional info"))
        f.write("\n\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.close()
        super().on_train_end(logs)


folderName = "logs/{}".format(time())
tensorboard = TBLog(folderName)

filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
with tf.device('/gpu:0'):
    model.fit(x, y,
              batch_size=128,
              epochs=epochsToTrain,
              verbose=2,
              callbacks=[generate_text, checkpoint, tensorboard])
