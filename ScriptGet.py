from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

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


#type 0=scenetransition,1=dialogue
class Line:
    def __init__(self,type,prefix,text):
        self.type=type
        self.prefix=prefix
        self.text=text

friends_root_url='https://fangj.github.io/friends/'
raw_html=simple_get(friends_root_url)
print(len(raw_html))

html=BeautifulSoup(raw_html,'html.parser')
for a in html.select('a'):
    episode_url=friends_root_url+a['href']
    print("Getting Episode {0} [from {1}]".format(a.text,episode_url))
    episode_raw_html=simple_get(episode_url)
    episode_html=BeautifulSoup(episode_raw_html,'html.parser')
    episode_html_lines=episode_html.select('p')
    lines=[]

    for i,p in enumerate(episode_html_lines):
        if(i>1 and i<len(episode_html_lines)-1):
            #print("****NEW LINE****")
            #if direction
            if(p.text[0]=="["):
                #print("DIRECTION")
                #print(p.text)
                lines.append(Line(0, "SCENE", p.text))

            else:
                pText=p.text
                if(pText.find("[")>-1):
                    #print("NO NEW LINE PROBLEM")
                    bracketIndex=p.text.find("[")
                    direction=pText[bracketIndex:]

                    lines.append(Line(0, "SCENE", pText[bracketIndex:]))
                    pText=pText[:bracketIndex]

                bolded=p.select('b')+p.select('strong')
                if(len(bolded)>0):
                    #print("DIALOGUE")
                    for name in bolded:
                        #print(name.text[:-1])
                        #print("_____")
                        #print(pText[len(name.text):])
                        lines.append(Line(1,name.text[:-1],pText[len(name.text):]))
                        break;

    break;
print(len(lines))
