import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#importar dataframe

df = pd.read_json("03092020_2kcomments.json")

#EDA

#lowercases

df = df.applymap(lambda x: x.lower())

#separar em lista e retirar pontuação
retirarPontos = lambda x: x.lower().replace(",","").replace(".","").replace(")","").replace("(","").replace("!","").replace("?","").split()

df = df.applymap(retirarPontos).applymap(lambda x: " ".join(x))

#wordcloud

from wordcloud import WordCloud


#wordcloud = WordCloud(width=3000,height=2000,background_color="black",stopwords=stopwords).generate(str(df[0].values).replace("'",""))

#plotar o gráfico

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

df["sem stopwords"] = df.applymap(stopRemove)
#df["sem stopwords"] = df["sem stopwords"].apply(lambda x: " ".join(x))

#Criar distribuição de frequencias, juntar todas strings em uma só string

stringTotal = []
for i, x in df.iterrows():
    stringTotal = stringTotal + x["sem stopwords"]

#stringTotal = " ".join(stringTotal)

from nltk.probability import FreqDist

Freq = FreqDist(stringTotal)
#plt.title("Distribuição de Frequências de Ocorrência das palavras")
plt.style.use('dark_background')

fig2 = plt.figure(figsize = (40,30))
Freq.plot(30,cumulative=False,color="red")
fig2.savefig("teste.png")

#Tamanho das frases - 

tamanho = df[0]

tamanho = tamanho.apply(lambda x: len(x.split(" ")))
tamanho.plot(cmap="viridis")


#analise

#Data Lemmatization

import joblib
folder = 'trained_POS_taggers/' 
tagger = joblib.load(folder+'POS_tagger_brill.pkl')

df["pos_tag"] = df["sem stopwords"].apply(lambda x: tagger.tag(x))





#converter dataframe em um corpus

#def CreateCorpusFromDataFrame(corpusfolder,df):
#    for index, r in df.iterrows():
#        body=r[0]
#        fname=str(index)+'.txt'
#        corpusfile=open(corpusfolder+'/'+fname,'a')
#        corpusfile.write(str(body))
#        corpusfile.close()

