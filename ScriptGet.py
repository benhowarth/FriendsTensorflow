from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import random

def simple_get(url):
    try:
        with closing(get(url,stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error("Error during requests to {0} : {1}".format(url,str(e)))
        return None

def is_good_response(resp):
    content_type=resp.headers['Content-Type'].lower()
    return(resp.status_code==200 and content_type is not None and content_type.find('html')>-1)

def log_error(e):
    print(e)



characters=[]
def getCharacterId(name):
    for char in characters:
        if(char.name==name):
            return char.id
    return -1
class Character:
    def __init__(self,name):
        self.name=name
        self.id=len(characters)
    def __str__(self):
        return "CHARACTER {0}: '{1}'".format(self.id,self.name)

tokens={}
detokens={}
def toke(str):
    tokensToReturn=[]
    for c in str:
        if c in tokens:
            tokensToReturn.append(tokens[c])
        else:
            tokens[c]=len(tokens)
            detokens[len(tokens)]=c
            tokensToReturn.append(tokens[c])

    return tokensToReturn
    #for char in characters:
    #    if (line.text.find(char.name) > -1):
    #        line.charIds.append(char.id)
def detoke(tokensToDetoke):
    res=""
    for t in tokensToDetoke:
        res=res+detokens[t+1]
    return res

lineTypes=["MISC","SCENE","DESCRIPTION","DIALOGUE"]
class Line:
    def __init__(self,type,charIds,text):
        self.type=type
        self.charIds=charIds
        self.text=text.lower()
        self.tokens=toke(text)
    def __str__(self):
        return  "|{0}| characters: {1} '{2}'".format(lineTypes[self.type],self.charIds,self.text)


def addLine(lineType,charList,text):
    charIds=[]
    for char in charList:
        char=char.lower()
        charId=getCharacterId(char)
        if(charId==-1):
            newChar=Character(char)
            characters.append(newChar)
            charId=newChar.id
        charIds.append(charId)

    lineToAdd=Line(lineType,charIds,text)
    #print(lineToAdd)
    lines.append(lineToAdd)





def analyseAndAddLineStrings(lineStrings):

    lineStrings=list(filter(None,lineStrings))

    print("Reading {0} lines".format(len(lineStrings)))
    for line in lineStrings:
        #print("NEXT LINE")
        #print("\t"+line)
        charList=[]
        if(line[0]=="["):
            #print("SCENE")
            lineType=1
            text=line[1:-1]
        elif(line[0]=="("):
            #print("DESCRIPTION")
            lineType=2
            text=line[1:-1]
        elif(line.find(":")>-1):
            #print("DIALOGUE")
            lineType=3
            text=line[line.find(":")+1:]
            charNames=line[:line.find(":")]
            charList=charNames.split(", ")
            for c in range(len(charList)):
                charString=charList[c]
                andIndex=charString.find("and ")
                if(andIndex>-1):
                    charList[c]=charString[andIndex+4:]
        else:
            lineType=0
            text=line
        addLine(lineType,charList,text)




friends_root_url='https://fangj.github.io/friends/'
raw_html=simple_get(friends_root_url)
#print(len(raw_html))
epCount=0
lines=[]
html=BeautifulSoup(raw_html,'html.parser')
for a in html.select('a'):
    episode_url=friends_root_url+a['href']
    print("Getting Episode {0} [from {1}]".format(a.text,episode_url))
    episode_raw_html=simple_get(episode_url)
    episode_html=BeautifulSoup(episode_raw_html,'html.parser')
    episode_html_lines=episode_html.select('p')
    lineStrings = []

    allEpisodeString=""
    for i,p in enumerate(episode_html_lines):
        if(i>1 and i<len(episode_html_lines)-1):
            formattedP=p.text.replace("\n"," ")
            formattedP.replace("  "," ")
            squareBracketIndex=formattedP.find("[")
            if(squareBracketIndex>-1):
                lineStrings.append(formattedP[:squareBracketIndex])
                lineStrings.append(formattedP[squareBracketIndex:])
            else:
                lineStrings.append(formattedP)

    analyseAndAddLineStrings(lineStrings)
    epCount+=1
    #uncomment for only one episode
    #break;

    if(epCount==24):
        break


for line in lines:
    if(line.type==1 or line.type==2):
        for char in characters:
            if(line.text.find(char.name)>-1):
                line.charIds.append(char.id)

for char in characters:
    print(char)

#for line in lines:
    #print(line)

print("Lines: {0}".format(len(lines)))

characterToTrain="joey"

characterToTrainId=getCharacterId(characterToTrain)
lines_to_learn=list(filter(lambda x:(characterToTrainId in x.charIds)and(x.type==3),lines))


just_text=list(map(lambda x:x.text,lines_to_learn))


just_text_string="".join(just_text)

word_ends=[".","?","!",":",";","(",")"," ","[","]","'","/",","]
wordList=[]

def splitPunc(word):
    if(len(word)>1):
        for end in word_ends:
            #punctuation index
            pi=word.find(end)
            if(pi>-1):
                if(len(word[:pi])==0):
                    return [word[pi]]+splitPunc(word[pi+1:])
                elif(len(word[pi+1:])==0):
                    return splitPunc(word[:pi])+[word[pi]]
                else:
                    return splitPunc(word[:pi])+[word[pi]]+splitPunc(word[pi+1:])

    return [word]

wordList=splitPunc(just_text_string)

        
chars=sorted(list(set(wordList)))
#chars=sorted(list(set(just_text_string)))

char_indices = dict((c,i) for i,c in enumerate(chars))
indices_char = dict((i,c) for i,c in enumerate(chars))

maxlen=15
step=5
sentences=[]
next_chars=[]

for i in range(0,len(wordList)-maxlen,step):
    sentences.append(wordList[i:i+maxlen])
    next_chars.append(wordList[i+maxlen])

#print(sentences[:10],"\n")
#print(next_chars[:10])
##lets do some neural net

import tensorflow as tf
import numpy as np

x=np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
y=np.zeros((len(sentences),len(chars)),dtype=np.bool)

for i,sentence in enumerate(sentences):
    for t,char in enumerate(sentence):
        x[i,t,char_indices[char]]=1
    y[i,char_indices[next_chars[i]]]=1

    
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
import sys
import io
from time import time
from keras.callbacks import TensorBoard


model=Sequential()
model.add(LSTM(128,input_shape=(maxlen,len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

lr=0.04
optimizer=RMSprop(lr=lr)
model.compile(loss='categorical_crossentropy',optimizer=optimizer)

epochsToTrain=10


def sample(preds,temperature=1.0):
    preds=np.asarray(preds).astype('float64')
    preds=np.log(preds)/temperature
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1,preds,1)
    return np.argmax(probas)

diversities=[0.2,0.5,1.0,1.2]
sentenceLenToGen=60
def on_epoch_end(epoch,logs):
    if epoch+1==1 or epoch+1==epochsToTrain:
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        start_index=random.randint(0,len(wordList)-maxlen-1)
        stringToFile="EPOCH "+str(epoch)+"\n\n"
        for diversity in diversities:
            print('diversity:',diversity)
            generated=''
            sentence=wordList[start_index:start_index+maxlen]
            generated+="".join(sentence)
            sys.stdout.write(generated)
            for i in range(sentenceLenToGen):
                x_pred=np.zeros((1,maxlen,len(chars)))
                for t,char in enumerate(sentence):
                    x_pred[0,t,char_indices[char]]=1.
                    
                preds=model.predict(x_pred,verbose=0)[0]
                next_index=sample(preds,diversity)
                next_char=indices_char[next_index]

                generated+=next_char
                sentence=sentence[1:]
                sentence.append(next_char)
                
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
            stringToFile+="DIVERSITY: "+str(diversity)+"\n"+generated+"\n\n\n"
        f=open("{}/output.txt".format(folderName),"a")
        f.write(stringToFile)
        f.close()
    else:
        print()
        print('----- Not generating text after Epoch: %d' % epoch)

generate_text = LambdaCallback(on_epoch_end=on_epoch_end)



params={}
params["lr"]=lr
params["diversities"]=",".join([str(d) for d in diversities])
params["sentenceLenToGen"]=sentenceLenToGen
params["maxlen"]=maxlen
params["step"]=step
params["epochsToTrain"]=epochsToTrain
params["characterToTrain"]=characterToTrain
params["characterToTrainId"]=characterToTrainId
params["epCount"]=epCount


#hyperparameters = [tf.convert_to_tensor([k, str(v)]) for k, v in params.items()]
import json
hyperparameters=json.dumps(params)
#print(hyperparameters)
#file_writer=tf.summary.FileWriter("logs/{}".format(time()))
#for k,v in params.items():
#    summary=tf.summary.text(k,tf.convert_to_tensor(str(v)))
#    file_writer.add_summary(summary).eval()
#file_writer.flush()
#file_writer.close()

class TBLog(TensorBoard):
    def __init__(self,log_dir):
        #self.hyperparams=params
        #self.writer=
        super().__init__(log_dir=log_dir)

    def on_train_end(self,logs=None):
        #for k,v in self.hyperparams.items():    
            #logs.update({k:v})
            #summary=tf.summary.text(k,tf.convert_to_tensor(str(v)))
            #text_tensor=tf.make_tensor_proto(v)
            #summary=tf.Summary()
            #summary.value.add(tag=str(k),tensor=text_tensor)
            #self.writer.add_summary(summary)
        #self.writer.flush()
        f=open("{}/params.txt".format(folderName),"w+")
        f.write(hyperparameters)
        f.close()
        super().on_train_end(logs)


folderName="logs/{}".format(time())
tensorboard = TBLog(folderName)



filepath="weights.hdf5"
checkpoint=ModelCheckpoint(filepath,
                           monitor='loss',
                           verbose=1,
                           save_best_only=True,
                           mode='min')
with tf.device('/gpu:0'):
    model.fit(x,y,
              batch_size=128,
              epochs=epochsToTrain,
              verbose=2,
              callbacks=[generate_text,checkpoint,tensorboard])
        
print(hyperparameters)
