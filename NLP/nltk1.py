##from nltk.tokenize import sent_tokenize, word_tokenize
##
##str="Hello, How are you today Mr. Adi? I am feeling rather sleepy"
##print(sent_tokenize(str))
##
##for i in word_tokenize(str):
##    print (i)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

str="Hello, How are you today Mr. Adi? I am feeling rather sleepy"
##stop_words=set(stopwords.words("english"))
words=word_tokenize(str)
##filteredWords=[]
##for w in words:
##    if w not in stopwords.words():
##        filteredWords.append(w)
##print(filteredWords)

filteredWords=[w for w in words if not w in stopwords.words()]

print(filteredWords)