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

for line in lines:
    print(line)

print("Lines: {0}".format(len(lines)))
