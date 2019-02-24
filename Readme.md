# Getting started with Machine Learning

## Day 1 : Feb 20 , 2019
 
 **Abstraction** : The NLTK module is a massive tool kit, aimed at helping you with the entire Natural Language Processing (NLP) methodology. NLTK will aid you with everything from splitting sentences from paragraphs, splitting up words, recognizing the part of speech of those words, highlighting the main subjects, and then even with helping your machine to understand what the text is all about. In this series, we're going to tackle the field of opinion mining, or sentiment analysis.

**What I leant** : learning how to do sentiment analysis with NLTK. Tokenizing word and sentences.

**Thoughts** : 

**Commit:**  [Link](https://github.com/IdeaLaboratory/MachineLearning/commit/fc790f9b17c442e08c028cd51680035be34acb0d)

## Day 2 : 

**What I leant** : stemming is kind of normalization.

**Thoughts** :

  *PorterStemmer algorithm does not follow linguistics rather a set of 05 rules for different cases that are applied in phases (step by step) to  generate stems**.
   This is the reason why PorterStemmer does not often generate stems that are actual English words. 
  It does not keep a lookup table for actual stems of the word but applies algorithmic rules to generate stems.
   It uses the rules to decide whether it is wise to strip a suffix.
  **advantage** PorterStemmer is known for its simplicity and speed.
```python
from nltk.stem import PorterStemmer
```

*The LancasterStemmer (Paice-Husk stemmer) is an iterative algorithm with rules saved externally.
 One table containing about 120 rules indexed by the last letter of a suffix.
 On each iteration, it tries to find an applicable rule by the last character of the word.
 Each rule specifies either a deletion or replacement of an ending. If there is no such rule, it terminates.
 It also terminates if a word starts with a vowel and there are only two letters left or if a word starts with a consonant and there are only       three characters left
 **advantage** LancasterStemmer is simple, but heavy stemming due to iterations and over-stemming may occur. Over-stemming causes the stems to be not linguistic, or they may have no meaning.

```python
from nltk.stem import LancasterStemmer
```
**Commit:**  [Link](https://github.com/IdeaLaboratory/MachineLearning/commit/90dab19decb26939ddc5ea807d62e26e2c6b98a7)

## Day 3 : 
 
**What I leant** : Parts of speech (POS) are noun, verb, adjective, adverb, pronoun, preposition, conjunction, interjection, and sometimes numeral, article, or determiner.
Learnt reading a text file and tag each word.

**Thoughts** : 
steps:

imports
 ```python
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union
```
Retrieve sentences from file and words from sentence.
apply pos_tag on words

```python
taggedWords = nltk.pos_tag(words)
```

note: PunktSentenceTokenizer is a pre trained but can be re-trained if require.

**Commit:**  [Link](https://github.com/IdeaLaboratory/MachineLearning/commit/11e40fe0407bc4c01c0b63a5607e80869f251ecc)