import nltk
import random
from nltk.corpus import movie_reviews

doc = []
for category in movie_reviews.categories():
	for field in movie_reviews.fileids(category):
		lst = list(zip(movie_reviews.words(field),category))
		doc.append(lst)
		
# print(doc[1])

random.shuffle(doc)
all_words =[]

for w in movie_reviews.words():
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(10))