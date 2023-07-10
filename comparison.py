import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from math import log, sqrt
import matplotlib.pyplot as plt
from wordcloud import WordCloud

user = pd.read_csv('output.csv', encoding = 'latin-1')
critic = pd.read_csv('all_games.csv', encoding = 'latin-1')

#text = " ".join(i for i in user.Comment)
#stopword = stopwords.words('english')
#wordcloud = WordCloud(stopwords=stopword, background_color="white").generate(text)
#plt.figure( figsize=(15,10))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.show()

#test = user.loc[user['Title'] == "The Legend of Zelda: Ocarina of Time"]
beforeAvg = user.groupby('Title')['Rating'].mean()
after = user.loc[user['Label'] == 1]
afterAvg = after.groupby('Title')['Rating'].mean()
#avg = test['Rating'].mean()
print(beforeAvg)
print(afterAvg) #+ " User Average: " + avg)

criticAvg = critic.groupby('name')['metascore'].mean()/10
print(criticAvg)

zeldaBU = user.loc[user['Title'] == "The Legend of Zelda: Ocarina of Time"]
zeldaBU = zeldaBU['Rating'].mean()
zeldaAU = user.loc[user['Title'] == "The Legend of Zelda: Ocarina of Time"]
zeldaAU = zeldaAU.loc[zeldaAU['Label'] == 1]
zeldaAU = zeldaAU['Rating'].mean()
zeldaC = critic.loc[critic['name'] == "The Legend of Zelda: Ocarina of Time"]
zeldaC = zeldaC['metascore'].mean()/10

print("Zelda: Ocarina of Time Before User Score:")
print(zeldaBU)
print("Zelda: Ocarina of Time After User Score: ")
print(zeldaAU)
print("Zelda: Ocarina of Time Critic Score: ")
print(zeldaC)

haloBU = user.loc[user['Title'] == "Halo 5: Guardians"]
haloBU = haloBU['Rating'].mean()
haloAU = user.loc[user['Title'] == "Halo 5: Guardians"]
haloAU = haloAU.loc[haloAU['Label'] == 1]
haloAU = haloAU['Rating'].mean()
haloC = critic.loc[critic['name'] == "Halo 5: Guardians"]
haloC = haloC['metascore'].mean()/10

print("Halo 5 Before User Score:")
print(haloBU)
print("Halo 5 After User Score: ")
print(haloAU)
print("Halo 5 Critic Score: ")
print(haloC)