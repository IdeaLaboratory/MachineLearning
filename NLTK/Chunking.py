import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
import os


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
        
        grammer = R"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>*}"""
        chunckingParser = nltk.RegexpParser(grammer)
        chunk = chunckingParser.parse(taggedWords)

        #for tWord in taggedWords:
        chunk.draw()

# calling the function            
partOfSpeechTaggig()