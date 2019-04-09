from ScriptGet import *
import random


characterToTrain = "joey"

characterToTrainId = getCharacterId(characterToTrain)
#lines_to_learn = list(filter(lambda x: (characterToTrainId in x.charIds) and (x.type == 3), lines))
lines_to_learn = lines

#just_text = list(map(lambda x: x.text, lines_to_learn))

#just_text_string = "".join(just_text)

word_ends = [".", "?", "!", ":", ";", "(", ")","-", " ", "[", "]", "'", "/", ","]


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

maxlen = 16
step = 4
charactersInvolved=[]
sentences = []
next_chars = []

for line in lines_to_learn:
    #wordList=splitPunc(line.text)
    wordList=[c for c in line.text]
    if(len(wordList)<maxlen):
        wordList=wordList+[" " for i in range(maxlen-len(wordList))]
    for i in range(0, len(wordList) - maxlen, step):
        charactersInvolved.append(line.charIds)
        sentences.append(wordList[i:i + maxlen])
        next_chars.append(wordList[i + maxlen])

flattened_sentences= [y for x in sentences for y in x]
flattened_sentences+= next_chars
chars = sorted(list(set(flattened_sentences)))
# chars=sorted(list(set(just_text_string)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

import tensorflow as tf
import numpy as np


#get all lines
#split into sentences {text:"20 words",charIds:[0,1]}
#

x = np.zeros((len(sentences), maxlen, len(chars)+len(characters)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)+len(characters)), dtype=np.bool)

for i,characterInvolved in enumerate(charactersInvolved):
    for t in range(maxlen):
        x[i, t, characterInvolved] = 1

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, len(characters)+char_indices[char]] = 1
    y[i, len(characters)+char_indices[next_chars[i]]] = 1

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
import sys
import io
from time import time
from keras.callbacks import TensorBoard

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars)+len(characters))))
model.add(Dense(len(chars)+len(characters)))
model.add(Activation('softmax'))

lr=0.01
optimizer = RMSprop(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

epochsToTrain = 60



def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

diversities = [0.2, 0.5, 1.0, 1.2]
sentenceLenToGen = 60


structurePairs=[
    [getCharacterId("SCENE"),60],
    [getCharacterId("joey"),60],
    [getCharacterId("chandler"),80],
    [getCharacterId("monica"),100]
]

print(structurePairs)



def on_epoch_end(epoch, logs):
    #if epoch + 1 == 1 or epoch + 1 == epochsToTrain:
    if epoch + 1 == 1 or epoch + 1 == epochsToTrain or (epoch+1)%10==0:
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        stringToFile = "EPOCH " + str(epoch) + "\n"
        stringToFile+=str(logs)+"\n\n"
        for diversity in diversities:
            print('diversity:', diversity)
            stringToFile += "DIVERSITY: " + str(diversity) + "\n"
            for line in structurePairs:
                generated = ''
                start_index = random.randint(0, len(sentences) - maxlen - 1)
                while(not (line[1] in charactersInvolved[start_index])):
                    start_index = random.randint(0, len(sentences) - maxlen - 1)

                sentence = sentences[start_index]



                generated += "".join(sentence)
                sys.stdout.write(getCharacterName(line[0])+": ")
                sys.stdout.write(generated)
                for i in range(line[1]):
                    x_pred = np.zeros((1, maxlen, len(characters)+len(chars)))
                    for t, char in enumerate(sentence):
                        x_pred[0, t, len(characters)+char_indices[char]] = 1.
                        #ensure one character is speaking the line
                        x_pred[0,t,line[0]] = 1.
                    preds = model.predict(x_pred, verbose=0)[0]
                    next_index = sample(preds, diversity)-len(characters)
                    try:
                        next_char = indices_char[next_index]
                    except KeyError:
                        print("!ERROR!")
                        next_char=indices_char[random.randint(0,len(indices_char))]
                    generated += next_char
                    sentence = sentence[1:]
                    sentence.append(next_char)

                    sys.stdout.write(next_char)
                    sys.stdout.flush()
                print()
                stringToFile+= getCharacterName(line[0])+"len("+str(line[1])+")" + generated+"\n"
            stringToFile+="\n\n\n"
        f = open("{}/output.txt".format(folderName), "a")
        f.write(stringToFile)
        f.close()
    else:
        print()
        print('----- Not generating text after Epoch: %d' % epoch)


generate_text = LambdaCallback(on_epoch_end=on_epoch_end)

params = {}
params["lr"] = str(lr)
params["diversities"] = ",".join([str(d) for d in diversities])
params["sentenceLenToGen"] = str(sentenceLenToGen)
params["maxlen"] = str(maxlen)
params["step"] = str(step)
params["epochsToTrain"] = str(epochsToTrain)
#params["characterToTrain"] = characterToTrain
#params["characterToTrainId"] = str(characterToTrainId)
params["epCount"] = str(epCount)


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

    def on_train_begin(self, logs=None):
        f = open("{}/params.txt".format(folderName), "a")
        f.write(dict_to_table(params, "Hyperparameters & additional info"))
        f.write("\n\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.close()
        super().on_train_begin(logs)


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
