import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
import os

# nltk.help.upenn_tagset()  #=> POS tag list:

# CC	coordinating conjunction
# CD	cardinal digit
# DT	determiner
# EX	existential there (like: "there is" ... think of it like "there exists")
# FW	foreign word
# IN	preposition/subordinating conjunction
# JJ	adjective	'big'
# JJR	adjective, comparative	'bigger'
# JJS	adjective, superlative	'biggest'
# LS	list marker	1)
# MD	modal	could, will
# NN	noun, singular 'desk'
# NNS	noun plural	'desks'
# NNP	proper noun, singular	'Harrison'
# NNPS	proper noun, plural	'Americans'
# PDT	predeterminer	'all the kids'
# POS	possessive ending	parent's
# PRP	personal pronoun	I, he, she
# PRP$	possessive pronoun	my, his, hers
# RB	adverb	very, silently,
# RBR	adverb, comparative	better
# RBS	adverb, superlative	best
# RP	particle	give up
# TO	to	go 'to' the store.
# UH	interjection	errrrrrrrm
# VB	verb, base form	take
# VBD	verb, past tense	took
# VBG	verb, gerund/present participle	taking
# VBN	verb, past participle	taken
# VBP	verb, sing. present, non-3d	take
# VBZ	verb, 3rd person sing. present	takes
# WDT	wh-determiner	which
# WP	wh-pronoun	who, what
# WP$	possessive wh-pronoun	whose
# WRB	wh-abverb	where, when

#read from txt file

# inputText = state_union.raw(os.path.abspath(os.path.join(os.getcwd(),"..\Dataset\RabindranathTagore.txt")))
inputText = state_union.raw(os.path.abspath(os.getcwd()+"\Dataset\RabindranathTagore.txt"))
experimentText = state_union.raw("I:\Information\WorkSpace\AdiRepo\MachineLearning\DataSet\SubhasChandraBose.txt")    

#train tokenizer, if require.
trainedTokenizer = PunktSentenceTokenizer()
# trainedTokenizer = PunktSentenceTokenizer(inputText)

#tokenizing experimentText
sentences =trainedTokenizer.tokenize(experimentText)

def partOfSpeechTaggig():
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)

        #  pos_tag() take list of words or sentence as input and tag part of speech
        taggedWords = nltk.pos_tag(words)
        #for tWord in taggedWords:
        print(taggedWords)

# calling the function            
partOfSpeechTaggig()