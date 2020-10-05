import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
from textwrap import wrap

from wordcloud import WordCloud

df = pd.read_json("03092020_2kcomments.json")

#Criar nova coluna contendo instâncias separadas em um array e sem pontuação
retiraPontosSeparar = lambda x: x.lower().replace(",","").replace(".","").replace(")","").replace("(","").split()
retiraPontos = lambda x: x.lower().replace(",","").replace(".","").replace(")","").replace("(","")

df["wordMap"] = df.applymap(retiraPontos)
df["array"]=df[0].apply(retiraPontosSeparar)
df.head()

#criar um dataframe com todas as palavras, vou usar o intertools
import itertools

listOfWord = list(itertools.chain.from_iterable(list(df["array"].values)))

#criar dataframe com todas as palavras

wordsDF = pd.DataFrame(listOfWord)
wordsDF.head()
pd.DataFrame(wordsDF[0].value_counts())
#filtrar preposição, artigo e pronome
def retirarString(x,n=1):
    if(len(x)<=n):
        return np.nan
    else:
        return x

def menosUm(x):
    if(x<=4):
        return np.nan
    else:
        return x
    
retirarString("i")
CU=wordsDF.applymap(lambda x: retirarString(x,5))[0]
#.value_counts().apply(lambda x: menosUm(x)).dropna()

#wc = WordCloud(width=400, height=300, max_words=150,colormap="Dark2").generate_from_frequencies(df)



#TESTE 2 ======================================================

filtro1 = lambda X: " ".join([x if len(x)>=4 else "" for x in X.split(" ")])

def generate_wordcloud(data,title):
  wc = WordCloud(width=400, height=330, max_words=150,colormap="Dark2").generate(data)
  plt.figure(figsize=(10,8))
  plt.imshow(wc, interpolation='bilinear')
  plt.axis("off")
  plt.title('\n'.join(wrap(title,60)),fontsize=13)
  plt.show()

filtro = df["wordMap"]
filtro = pd.DataFrame(filtro.apply(filtro1))
filtro = filtro.applymap(lambda x: x.replace("  "," "))