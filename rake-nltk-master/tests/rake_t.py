from rake_nltk import Rake

from difflib import SequenceMatcher

import nltk

#from nltk.stem import PorterStemmer

#from nltk.stem import LancasterStemmer

from nltk.stem import WordNetLemmatizer 
  
from nltk.corpus import wordnet

import gensim

from nltk.tokenize import sent_tokenize, word_tokenize

import array 

"""porter = PorterStemmer()

lancaster=LancasterStemmer()

#defining stemming for sentence

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)
"""


#x=stemSentence(sentence)

lemmatizer = WordNetLemmatizer()

# function to convert nltk tag to wordnet tag

#for lemmatization
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)
 
#defining similar

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

a = Rake(min_length=1, max_length=4)
r = Rake(min_length=1, max_length=4)
s = Rake(min_length=1, max_length=4)
t = Rake(min_length=1, max_length=4)
u = Rake(min_length=1, max_length=4)
v = Rake(min_length=1, max_length=4)
# Extraction given the text.
str = open('answer.txt', 'r').read()

str1 = open('key1.txt','r').read()
str2 = open('key2.txt','r').read()
str3 = open('key3.txt','r').read()
str4 = open('key4.txt','r').read()
manual = [
            'rapid automatic keyword extraction', 'concrete application depend', 'widely use nlp technique', 'automatically extract keywords', 'compact representation', 'keywords', 'write', 'word', 'well', 'together', 'sequence', 'rake', 'purpose', 'provide', 'one', 'lot', 'language', 'known', 'domain', 'document', 'content', 'algorithm'
            ]
str5='.'.join(manual)

ans=lemmatize_sentence(str)

s1=lemmatize_sentence(str1)
s2=lemmatize_sentence(str2)
s3=lemmatize_sentence(str3)
s4=lemmatize_sentence(str4)
s5=lemmatize_sentence(str5)

#a=stemSentence(s1)
#b=stemSentence(s2)
a.extract_keywords_from_text(ans)
r.extract_keywords_from_text(s1)
s.extract_keywords_from_text(s2)
t.extract_keywords_from_text(s3)
u.extract_keywords_from_text(s4)
v.extract_keywords_from_text(s5)

# Extraction given the list of strings where each string is a sentence.
#r.extract_keywords_from_sentences("list of sentences")

# To get keyword phrases ranked highest to lowest.
a1=a.get_ranked_phrases()
r1=r.get_ranked_phrases()
r2=s.get_ranked_phrases()
r3=t.get_ranked_phrases()
r4=u.get_ranked_phrases()
r5=v.get_ranked_phrases()
#print(r.extract_keywords_from_sentences(s3))

#print(s.extract_keywords_from_sentences(s4))
# To get keyword phrases ranked highest to lowest with scores.
#print(similar(r.get_ranked_phrases_with_scores(),s.get_ranked_phrases_with_scores()))





#c=stemSentence(s5)
#print(answer_list)
#print(phrase_list)
a1.sort()
r1.sort()
r2.sort()
r3.sort()
r4.sort()
r5.sort()

arr = array.array('d', [0,0,0,0,0])
arr[0]=similar(a1,r1)
arr[1]=similar(a1,r2)
arr[2]=similar(a1,r3)
arr[3]=similar(a1,r4)
arr[4]=similar(a1,r5)

print(arr)
print("keywords from key1 \n")
print(r1,'\n')
print("keywords from key2 \n")
print(r2,'\n')
print("keywords from key3 \n")
print(r3,'\n')
print("keywords from key4 \n")
print(r4,'\n')
print("keywords from key5 \n")
print(r5,'\n')
print(max(arr))
#print(r.get_word_degrees(),'\n')

#print(r.get_ranked_phrases_with_scores())
#print(s.get_ranked_phrases_with_scores())


#print(s.get_word_degrees(),'\n')
#print(r._build_word_co_occurance_graph(str))
