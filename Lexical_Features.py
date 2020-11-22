#!/usr/bin/env python3

"""
Created on Sat Jul  6 13:57:17 2019

@author: mahparsa


"""
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import gutenberg
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim import utils 
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import stem_text
import collections
from collections import Counter
from stemming.porter2 import stem
from numpy import array
stop_words = set(stopwords.words('english')) 
#from nltk.stem import LancasterStemmer

from difflib import SequenceMatcher


lemmatizer = WordNetLemmatizer()

def percentage(count, total): 
     return 100 * count / total  

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

import matplotlib.pyplot as plt    
from nltk.corpus import stopwords  

def lexical_diversity(text): 
    return len(set(text)) / len(text)


def READ_INT( parameters ):
   "use the root and read files and make a list of that"
   corpus_root = parameters # Mac users should leave out C:
   corpus = PlaintextCorpusReader(corpus_root, '.*txt')  #
   doc = pd.DataFrame(columns=['string_values'])
   for filename in corpus.fileids(): 
       value1=corpus.raw(filename)
       doc = doc.append({'string_values': value1}, ignore_index=True)
   docs=doc.values.tolist()
   return [docs]


def Pre_Word( doc ):
    #provide a list of words.
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    m=str(doc)
    mm=m.lower()
    mmm=lemmatizer.lemmatize(mm)
    return [mmm]

def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors) 

docs_A=READ_INT('New_Data_SZ')

def Stat(docs_A): 
    #measure similarity between sentenses
    Stat_w=[]
    Stat_s=[]
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k]) #string
        Sent_doc=sent_tokenize(doc) #list
        Stat_s=np.append(Stat_s, len(Sent_doc) )
        W=word_tokenize(doc)
        W=[t for t in W if t.isalpha()]
        Stat_w=np.append(Stat_w, len(W) )
    return[Stat_s, Stat_w, sum(Stat_s), sum(Stat_w) ]
def Coherence_M_1(docs_A): 
    #measure the lexical richness
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        Co_2=lexical_diversity(doc)
        Coh_M=np.append(  Coh_M,  Co_2)
    return[Coh_M] 

#    
##Brunet Index
#
def Coherence_M_2(docs_A): 
    stop_words = set(stopwords.words('english')) 
    #Measure the Brunet Index
    #after removing stop words 
    stemmer = nltk.PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    Lexical_BI=np.array([])
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        doc=doc.lower()
        doc=lemmatizer.lemmatize(doc)
        Sent_doc=sent_tokenize(doc)
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        #word_tokens = [w for w in tokenized_word if w.isalpha() and not in stop_words ] 
        word_tokens = [w for w in tokenized_word if w.isalpha() and not w in stop_words  ] 
        
        BI_N = [stemmer.stem(w) for w in  word_tokens]
        #BI_N = word_tokens
        #BI_U = sorted (set([stemmer.stem(verb) for verb in  word_tokens]))
        BI_U = sorted (set(BI_N))
        BI_F=len(BI_N) ** (len(BI_U) ** -0.165)
        Lexical_BI=np.append(Lexical_BI, BI_F)
        Coh_M=Lexical_BI/sum(Lexical_BI)
    return[Coh_M]             
#    
#Honore Satistic
        
def Coherence_M_3(docs_A): 
    #Measure Honore Satistic
    #after removing stop words 
    import math 
    stemmer = nltk.PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    Lexical_HS=[]
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        doc=doc.lower()
        doc=lemmatizer.lemmatize(doc)
        Sent_doc=sent_tokenize(doc)
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        word_tokens = [w for w in tokenized_word if w.isalpha() and not w in stop_words  ] 
        N = [stemmer.stem(verb) for verb in word_tokens]
        fdist = FreqDist(N)
        Words_freq =np.asarray([fdist[w] for w in  N ])
        One_Time_Words=np.asarray(np.where(Words_freq==1))
        h=np.array(N)
        N_1 =len(h[One_Time_Words])
        U = len(sorted (set([stemmer.stem(verb) for verb in  word_tokens])))
        Honore_Satistic_Index=(100*math.log10(len(N)))/(1-((N_1)/U))
        #math.log10(N)
        Lexical_HS=np.append(Lexical_HS, Honore_Satistic_Index)
        Coh_M=Lexical_HS/sum(Lexical_HS)
    return[Coh_M]                 

##------   
##Readability of Transcripts 
# 
##1.Flesch Reading Score
#
def Coherence_M_4(docs_A): 
    #Measure Readability 
    #after removing stop words 
    import syllables
    Factor_1=206.835
    Factor_2=1.015
    Factor_3=84.6
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        doc=str(MyDoc[k])
        doc=doc.lower()
        Sent_doc=sent_tokenize(doc)
        T_Sent=len( Sent_doc) #Total Sentenses
        T_Word=np.array([])
        T_Syll=np.array([])
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        word_tokens = [w for w in tokenized_word if w.isalpha() ] 
        Syllables_Word= [ syllables.estimate(w) for w in word_tokens ] 
        T_Syll= np.append(T_Syll, sum(Syllables_Word)) #Total Syllables
        T_Word=np.append(T_Word, len(word_tokens)) #Total words            
        M = Factor_1 - Factor_2* (sum(T_Word)/T_Sent) - Factor_3 * (sum(T_Syll)/sum(T_Word)) 
        Coh_M=np.append(  Coh_M,  M)
    return[Coh_M]                    
#2.Flesch-Kincaid Grade Level       

def Coherence_M_5(docs_A): 
    #Measure Flesch-Kincaid Grade Level  
    #after removing stop words 
    import syllables
    Factor_1=0.39
    Factor_2=11.8
    Factor_3=15.59
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        Sent_doc=sent_tokenize(doc)
        T_Sent=len( Sent_doc) #Total Sentenses
        T_Word=np.array([])
        T_Syll=np.array([])
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        word_tokens = [w for w in tokenized_word if w.isalpha() ] 
        Syllables_Word= [ syllables.estimate(w) for w in word_tokens ] 
        T_Syll= np.append(T_Syll, sum(Syllables_Word)) #Total Syllables
        T_Word=np.append(T_Word, len(word_tokens)) #Total words            
        M = Factor_1 * (sum(T_Word)/T_Sent) + Factor_2 * (sum(T_Syll)/sum(T_Word)) + Factor_3
        Coh_M=np.append(  Coh_M,  M)
    return[Coh_M]           
# 


No_Measure=5
No_Subject=14
h=np.zeros((No_Measure, No_Subject))
#for i in range(No_Measure):
    
h[0][:]=np.asarray(Coherence_M_1( docs_A[0]))
h[1][:]=np.asarray(Coherence_M_2( docs_A[0]))
h[2][:]=np.asarray(Coherence_M_3( docs_A[0]))
h[3][:]=np.asarray(Coherence_M_4( docs_A[0]))
h[4][:]=np.asarray(Coherence_M_5( docs_A[0]))

hh=np.transpose(h)

class_names=['Depression','Psychosis' ]
#dataset = pd.DataFrame({'Lexical_Diversity':hh[:,0],'Brunet_Index':hh[:,1],'Honore_Satistic':hh[:,2],'Flesch Reading':hh[:,3], 'Flesch-Kincaid':hh[:,4], 'Ambigous_Pronouns':hh[:,5],'First_Pronouns_Ratio':hh[:,6], 'Third_Pronouns_Ratio':hh[:,7], 'Noun_Verb_Ratio':hh[:,8], 'Noun_Ratio':hh[:,9],'Subordinate_Coordinate_Ratio':hh[:,12],'Propositional_Density':hh[:,11], 'Content_Density':hh[:,12],  'Tangentiality':hh[:,13], 'Incoherence_SA_1':hh[:,14],'Incoherence_SA_2':hh[:,15],'Incoherence_SA_3':hh[:,16] })                     
dataset = pd.DataFrame({'Lexical_Diversity':hh[:,0],'Brunet_Index':hh[:,1],'Honore_Satistic':hh[:,2],'Flesch Reading':hh[:,3], 'Flesch-Kincaid':hh[:,4]})                     
