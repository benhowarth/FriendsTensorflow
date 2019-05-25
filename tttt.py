

import numpy as np
from collections import Counter
from time import time
from keras.models import load_model
import os
import sys
from ScriptGet import *
sys.getsizeof(lines[0])

runName="1556625990.947603-lr_0.003-sampleLen_10-step_5-epCount_100-batch_128-dropout_0.5-LSTMSize_128-optimizer_adam-wordFreqCutoff_6_wordLevel_Stacked-Bi_DropoutBetween"
word_freq_cutoff=6
maxlen=10

folderName="scripts/{}/{}_sceneReadScene".format(runName,time())

try:
    os.mkdir("scripts/{}".format(runName))
except FileExistsError:
    print("Already done a run of this name.")
os.mkdir(folderName)

##DUPED CODE
def sample(preds, temperature=1.0):
    preds=preds[len(characters):]
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

lines_to_learn = lines

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


from collections import Counter

def doesLineContainOnlyMostCommon(listOfWords):
    for word in listOfWords:
        if word not in chars:
            return False
    return True

END_LINE_CHAR="|"

#get only most common words
word_counter=Counter([c for line in lines_to_learn for c in splitPunc(line.text)])
chars=sorted([k for k in word_counter if word_counter[k] > word_freq_cutoff]+[END_LINE_CHAR])
print("len chars"+str(len(chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


###END DUPED CODE




def genScript(structurePairs,diversities):
    structurePairs=[[[0],10]]+structurePairs
    stringToFile=""
    stringToFileLatex=""
    for diversity in diversities:
        #print('diversity:', diversity)
        stringToFile += "DIVERSITY: " + str(diversity) + "\n"
        firstLine=True
        for line in structurePairs:
            if(firstLine):
                seedSentence=splitPunc("central perk, everyone is there")
            else:
                #if scen/description/misc then sample last line of that type
                if(line[0][0]<3):
                    seedSentence=lastLineTextListScene[-maxlen:]
                else:
                    seedSentence=lastLineTextList[-maxlen:]

            sentence = seedSentence
            #store last line for next iteration
            lastLine=line
            #lastLineTextList=[]


            #reset generated
            generated = ''
            if(firstLine):
                generated += "".join(sentence)
                firstLine=False

            #get and write character names to left
            #get first char names
            charNamesForLine=getCharacterName(line[0][0])
            #if there are other characters, get them too
            for j,id in enumerate(line[0]):
                if(j>0):
                    charNamesForLine+=", "+getCharacterName(id)
            #write names to stdout
            #sys.stdout.write(charNamesForLine+": ")

            #sys.stdout.write(generated)


            next_char=""
            #while(True):
            #while(next_char!=END_LINE_CHAR):

            #for number of words required for the line
            while(True):
            #for i in range(line[1]):

                #initialise x pred as 0s
                x_pred = np.zeros((1, maxlen, len(characters)+len(chars)))

                #get number of words from last line to use in prediction
                numberOfWordsFromLastLine=max(maxlen-i,0)

                #for each word in sentence so far
                for t, char in enumerate(sentence):
                    try:
                        x_pred[0, t, len(characters)+char_indices[char]] = 1.
                    except KeyError:
                        print(sentence + " error with char at " + str(t) + "=" + char)
                        x_pred[0, t, len(characters) + char_indices[" "]] = 1.
                    #if current word from last line
                    if(t>=numberOfWordsFromLastLine):
                        lineToSample=lastLine
                    else:
                        lineToSample=line
                    #go through all character ids 'speasing' line
                    for id in lineToSample[0]:
                        #set their value to true
                        x_pred[0,t,id] = 1.



                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_index=next_index#-len(characters)
                indexCount=0
                while(next_index not in indices_char and indexCount<4):
                    print("trying another index "+str(next_index)+" not found")
                    next_index = sample(preds, diversity)#-len(characters)
                    indexCount+=1


                #if(next_char!=END_LINE_CHAR):
                if(END_LINE_CHAR not in generated+next_char):
                    next_char = indices_char[next_index]

                    # try:
                    #     next_char = indices_char[next_index]
                    # except KeyError:
                    #     print("!ERROR!")
                    #     next_char=indices_char[random.randint(0,len(indices_char)-1)]
                    generated += next_char
                    sentence = sentence[1:]
                    sentence.append(next_char)
                    lastLineTextList=sentence
                    if(END_LINE_CHAR not in lastLineTextList):
                        lastLineTextList=lastLineTextList[:-1]+[END_LINE_CHAR]

                    #get scene line
                    if(line[0][0]<3):
                        lastLineTextListScene=lastLineTextList
                    
                    #sys.stdout.write(next_char)
                    #sys.stdout.flush()
                else:
                    break
            #print()
            #stringToFile+= getCharacterName(line[0])+"len("+str(line[1])+")" + generated+"\n"
            stringToFile += charNamesForLine+ ": " + generated + "( SEED: "+"".join(seedSentence)+")"+"\n"
            stringToFileLatex += "\\textit{"+charNamesForLine+ ":} " + generated +"\\newline \n"
        #stringToFile+="\n\n\n"
            
        f = open("{}/DIV_{}.txt".format(folderName,diversity),"w+")
        stringToFile.replace(END_LINE_CHAR,"")
        f.write(stringToFile)
        f.close

        f = open("{}/DIV_{}_LATEX.txt".format(folderName,diversity),"w+")
        stringToFileLatex.replace(END_LINE_CHAR,"")
        f.write(stringToFileLatex)
        f.close

        
        stringToFile=""
        stringToFileLatex=""
        print("Diversity {} done!".format(diversity))



model = load_model('currentModel.hdf5')

genScript(structure_to_gen,[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.2,2.0])


