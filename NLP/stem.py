# **PorterStemmer algorithm does not follow linguistics rather a set of 05 rules for different cases that are applied in phases (step by step) to generate stems**.
#  This is the reason why PorterStemmer does not often generate stems that are actual English words. 
# It does not keep a lookup table for actual stems of the word but applies algorithmic rules to generate stems.
#  It uses the rules to decide whether it is wise to strip a suffix.
# ** PorterStemmer is known for its simplicity and speed.
from nltk.stem import PorterStemmer

# The LancasterStemmer (Paice-Husk stemmer) is an iterative algorithm with rules saved externally.
#  One table containing about 120 rules indexed by the last letter of a suffix.
#  On each iteration, it tries to find an applicable rule by the last character of the word.
#  Each rule specifies either a deletion or replacement of an ending. If there is no such rule, it terminates.
#  It also terminates if a word starts with a vowel and there are only two letters left or if a word starts with a consonant and there are only three characters left
#  ** LancasterStemmer is simple, but heavy stemming due to iterations and over-stemming may occur. Over-stemming causes the stems to be not linguistic, or they may have no meaning.
from nltk.stem import LancasterStemmer


from nltk.tokenize import word_tokenize
ps = PorterStemmer()
ls = LancasterStemmer()

words = ["google","googly", "googling","googled","googlified", "googlian"]

for w in words:
   print("{0:20}{1:20}{2:20}".format(w,ps.stem(w),ls.stem(w)))
