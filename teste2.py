import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#configurar gráficos
font_title = {'family': 'DejaVu Sans',
        'color':  'white',
        'weight': 'bold',
        'size': 26,
        }
plt.style.use('dark_background')


#importar dataframe

df = pd.read_json("03092020_2kcomments.json")

#EDA

#lowercases

df = df.applymap(lambda x: x.lower())

import string

#separar em lista e retirar pontuação
retirarPontos = lambda x: x.lower().replace(",","").replace(".","").replace(")","").replace("(","").replace("!","").replace("?","").replace("-"," ").replace("_"," ").replace('"','').replace("'","").split()

df = df.applymap(retirarPontos).applymap(lambda x: " ".join(x))

#contar palavras

df["tamanho"] = df[0].apply(lambda x: len(x))

#wordcloud

from wordcloud import WordCloud


wordcloud = WordCloud(width=3000,height=2000,background_color="black",stopwords=stopwords).generate(str(df[0].values).replace("'",""))

#plotar o wordcloud

#fig = plt.figure(figsize = (40, 30), facecolor = 'k', edgecolor = 'k')
#plt.imshow(wordcloud, interpolation = 'bilinear')
#plt.axis('off')
#plt.tight_layout(pad=0)
#fig.savefig("wordcloud.png")

#verificar distribuição de frequências e correlação das palavras para cada texto
#definir stopwords

from nltk.corpus import stopwords
stopwords = set(stopwords.words("portuguese"))

stopRemove = lambda x: [palavra for palavra in x.split(" ") if palavra not in stopwords]

df["sem stopwords"] = df[0].apply(stopRemove)
#df["sem stopwords"] = df["sem stopwords"].apply(lambda x: " ".join(x))

#Criar distribuição de frequencias, juntar todas strings em uma só string

stringTotal = []
for i, x in df.iterrows():
    stringTotal = stringTotal + x["sem stopwords"]

#stringTotal = " ".join(stringTotal)

from nltk.probability import FreqDist
from nltk.probability import ProbDistI


Freq = FreqDist(stringTotal)
#plt.title("Distribuição de Frequências de Ocorrência das palavras")


fig2 = plt.figure(figsize = (40,30))
Freq.plot(30,cumulative=False,color="red")
fig2.savefig("teste.png")

#Tamanho das frases - 

tamanho = df[0]

tamanho = tamanho.apply(lambda x: len(x.split(" ")))
tamanho.plot(cmap="viridis")



#analise

#polaridade



from textblob import TextBlob

#Analisar sentimento
df["polaridade"] = df["sem stopwords"].apply(lambda x: TextBlob(" ".join(x)).sentiment.polarity)

#criar serie para polaridade neutra, positiva, negativa

df["status"]=df["polaridade"].apply(lambda x: "neutro" if type(x)==np.float and x==0 else x)
df["status"]=df["status"].apply(lambda x: "negativo" if type(x)==np.float and x<0 else x)
df["status"]=df["status"].apply(lambda x: "positivo" if type(x)==np.float and x>0 else x)

#analisar wordcloud de cada sentiment


wordcloudPositivo = WordCloud(width=3000,height=2000,background_color="black",stopwords=stopwords).generate(str(df[df["status"]=="positivo"][0].values).replace("'",""))
""" fig1 = plt.figure(figsize = (40, 30), facecolor = 'k', edgecolor = 'k')
plt.imshow(wordcloudPositivo, interpolation = 'bilinear')
plt.axis('off')
plt.title("Wordcloud - positivo")
plt.tight_layout(pad=0)
fig1.savefig("positivoWC.png")

wordcloudNeutro = WordCloud(width=3000,height=2000,background_color="black",stopwords=stopwords).generate(str(df[df["status"]=="neutro"][0].values).replace("'",""))
fig2 = plt.figure(figsize = (40, 30), facecolor = 'k', edgecolor = 'k')
plt.imshow(wordcloudNeutro, interpolation = 'bilinear')
plt.axis('off')
plt.title("Wordcloud - positivo")
plt.tight_layout(pad=0)
fig2.savefig("neutroWC.png")

wordcloudNegativo = WordCloud(width=3000,height=2000,background_color="black",stopwords=stopwords).generate(str(df[df["status"]=="negativo"][0].values).replace("'",""))
fig3 = plt.figure(figsize = (40, 30), facecolor = 'k', edgecolor = 'k')
plt.imshow(wordcloudNegativo, interpolation = 'bilinear')
plt.axis('off')
plt.title("Wordcloud - positivo")
plt.tight_layout(pad=0)
fig3.savefig("negativoWC.png") """

#Remover palavras que se repetem nos três diagramas 

stringTotalPositivo, stringTotalNeutro, stringTotalNegativo = [],[],[]






#analisar tamanho do texto X polaridade





def inList(list,element):
    try:
        list.index(element)
    except ValueError:
        return False
    else:
        return True

#verificar se há relação entre o tamanho das frases e o sentimento



sns.pairplot(df,hue="status",palette="husl")
plt.savefig("ParPlotSemFiltrar.png")



#muitos resultados neutros, descobri que o TextBlob não funciona muito bem para português
#traduzir para inglês
#from googletrans import Translator

#translator = Translator(service_urls=["translate.google.com"])
#translator.translate("string",dest="lan")
#translator.detect("text in here")