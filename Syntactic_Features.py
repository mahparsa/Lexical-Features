#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:51:21 2019

@author: mahparsa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:57:17 2019

@author: mahparsa

This aim to calculate the similairty between sentenses of an interview.
It has all features
"""
import nltk
import numpy as np
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import gutenberg
import pandas as pd
import nltk
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import nltk
nltk.download('wordnet')
import nltk
from gensim import utils 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize 
from gensim.parsing.preprocessing import stem_text
from nltk.stem import PorterStemmer
import collections
from collections import Counter
from stemming.porter2 import stem
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from numpy import array
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english')) 
#from nltk.stem import LancasterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher

from nltk.stem import WordNetLemmatizer

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
from nltk import sent_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity




def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors) 

docs_A=READ_INT('New_Data_SZ')
def Coherence_S_1(docs_A): 
    #ambigous pronouns
    #We can count how often a pronouns occurs in a text, and compute what percentage of the text is taken up by a specific pronouns
    #Measure the percentage of the third pronouns to all words. 
    Pronunce_words = set('he him his himself she her hers herself it its itself they them their theirs themselves'.split()) 

    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        Sent_doc=sent_tokenize(doc)
        tokenized_word=word_tokenize(doc)
        word_tokens = [w for w in tokenized_word if w.isalpha()] 
        Third_Pronouns = [w for w in word_tokens if w in Pronunce_words]
        U_Third_Pronouns=list(set(Third_Pronouns))
        U_word_tokens=list(set(word_tokens))
        Co_3=len(U_Third_Pronouns)/len(U_word_tokens)
        Coh_M=np.append(  Coh_M,  Co_3)
    return[Coh_M]    
#
 
def Coherence_S_2(docs_A): 
    #ratio of the first pronouns to all pronouns. 
    import pandas as pd    
    A_Pronunce_words = set('I my myself mine we us ourselves ours'.split()) 
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        tokenized_word=[]
        tokenized_word=word_tokenize(doc)
        word_tokens=[]
        word_tokens = [w for w in tokenized_word if w.isalpha()]
        tagg = pd.DataFrame()
        tagg=pd.DataFrame(nltk.pos_tag(word_tokens))
        Index_NP = list(np.where( tagg[1] == "PRP" ))
        First_Pronouns = [w for w in word_tokens if w in A_Pronunce_words]
        N_P=len( First_Pronouns)
        N_N = len(Index_NP)
        Co_5=N_P/N_N
        Coh_M=np.append(  Coh_M,  Co_5)
    return[Coh_M]   
       
def Coherence_S_3(docs_A): 
    #ratio of the third pronouns to names of persons. 
    import pandas as pd    
    A_Pronunce_words = set('he him his himself she her hers herself they them their theirs themselves'.split()) 
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        tokenized_word=[]
        tokenized_word=word_tokenize(doc)
        word_tokens=[]
        word_tokens = [w for w in tokenized_word if w.isalpha()]
        tagg = pd.DataFrame()
        tagg=pd.DataFrame(nltk.pos_tag(word_tokens))
        Index_NNP = list(np.where( tagg[1] == "NNP" ))
        Third_Pronouns = [w for w in word_tokens if w in A_Pronunce_words]
        N_P=len(Third_Pronouns)
        N_N = len(Index_NNP)
        Co_5=N_P/N_N
        Coh_M=np.append(  Coh_M,  Co_5)
    return[Coh_M]   



#3. The noun-verb ratio 
#..........................................................................        

def Coherence_S_4(docs_A): 
    
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    NV_Ratio_T=np.array([])
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        Sent_doc=sent_tokenize(doc)
        N_N=0
        N_P=0
        ii=0
        
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(Sent_doc[ii])
        word_tokens = [w for w in tokenized_word if w.isalpha()]
        tagg =[]
        tagg=pd.DataFrame(nltk.pos_tag(word_tokens))
            
            # NN	noun
        Index_NN = list  ( np.where( tagg[1] == "NN" ))  ####some error here
        Index_NNP = list ( np.where( tagg[1] == "NNP" ))  ####some error here
        Index_NNPS = list ( np.where( tagg[1] == "NNPS" ))  ####some error here
            
            #verb

        Index_VB = list  ( np.where( tagg[1] == "VB" ))  ####some error here
        Index_VBD = list ( np.where( tagg[1] == "VBD" ))  ####some error here
        Index_VBG = list ( np.where( tagg[1] == "VBG" ))  ####some error here
        Index_VBN = list ( np.where( tagg[1] == "VBN" ))  ####some error here
        Index_VBP = list ( np.where( tagg[1] == "VBP" ))  ####some error here
        Index_VBZ = list ( np.where( tagg[1] == "VBZ" )) 
        NV_Ratio= (len(Index_NNP)+len(Index_NN)+len(Index_NNPS))/(len(Index_VB)+len(Index_VBD)+len(Index_VBG)+len(Index_VBN)+ len(Index_VBP)+ len(Index_VBZ))
        NV_Ratio_T = np.append( NV_Ratio_T, NV_Ratio/len( Sent_doc))
        #M=sum(NV_Ratio_T)/len(Sent_doc) #I think it could be more useful. 
        
        Coh_M=NV_Ratio_T
    return[Coh_M]         


#
#The noun ratio 2
#..........................................................................        

def Coherence_S_5(docs_A): 
    #Measure the content density . 
    
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    N_Ratio_T=np.array([])
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        Sent_doc=sent_tokenize(doc)
        N_N=0
        N_P=0
        ii=0
        N_Ratio_T=np.array([])
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        word_tokens = [w for w in tokenized_word if w.isalpha()]
        tagg =[]
        tagg=pd.DataFrame(nltk.pos_tag(word_tokens))
            
            # NN	noun
        Index_NN = list  ( np.where( tagg[1] == "NN" ))  ####some error here
        Index_NNP = list ( np.where( tagg[1] == "NNP" ))  ####some error here
        Index_NNPS = list ( np.where( tagg[1] == "NNPS" ))  ####some error here
            
        N_Ratio= (len(Index_NNP)+len(Index_NN)+len(Index_NNPS))/(len( word_tokens))
        N_Ratio_T = np.append( N_Ratio_T, N_Ratio)
        
            
        #M=sum(NV_Ratio_T)/len(Sent_doc) I think it could be more useful. 
       # M=N_Ratio_T
        M=N_Ratio_T/len(Sent_doc)
        #M=sum(N_Ratio_T)/len(Sent_doc)
        Coh_M=np.append( Coh_M,  M)
    return[Coh_M] 
#
#
#5. The subordinate-coordinate ratio 2
def Coherence_S_6(docs_A):      
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    SC_Ratio_T=np.array([])
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        Sent_doc=sent_tokenize(doc)
        N_N=0
        N_P=0
        ii=0
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        word_tokens = [w for w in tokenized_word if w.isalpha()]
        tagg =[]
        tagg=pd.DataFrame(nltk.pos_tag(word_tokens))            
            # NN	noun
            #IN	preposition/subordinating conjunction
        Index_IN = list ( np.where( tagg[1] == "IN"))
            #CC	coordinating conjunction
        Index_CC = list ( np.where( tagg[1] == "CC"))
        SC_Ratio= (len(Index_IN)/len(Index_IN)+len(Index_CC))/len(Sent_doc)
        #M=sum(NV_Ratio_T)/len(Sent_doc) I think it could be more useful. 
        Coh_M=np.append(  Coh_M,  SC_Ratio )
    return[Coh_M] 

###1.1 New Propositional Density
###..........................................................................        
def Coherence_S_7(docs_A): 
    #Measure the propositional density and content density. 
    stop_words = set(stopwords.words('english')) 
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    Density_P=np.array([])
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        Sent_doc=sent_tokenize(doc)
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        word_tokens = [w for w in tokenized_word if w.isalpha()]
        word_tokens = [w for w in word_tokens if w.isalpha() and not w in stop_words ]   
        word_tokens = [stem(w) for w in  word_tokens]
        word_tokens =list(set(word_tokens))
        tagg =[]
        Index_NNP =[]
        Index_PRP =[]
        tagg=pd.DataFrame(nltk.pos_tag(word_tokens))
        Index_VB = list  ( np.where( tagg[1] == "VB" ))  ####some error here
        Index_VBD = list ( np.where( tagg[1] == "VBD" ))  ####some error here
        Index_VBG = list ( np.where( tagg[1] == "VBG" ))  ####some error here
        Index_VBN = list ( np.where( tagg[1] == "VBN" ))  ####some error here
        Index_VBP = list ( np.where( tagg[1] == "VBP" ))  ####some error here
        Index_VBZ = list ( np.where( tagg[1] == "VBZ" ))  ####some error here
            
            #adjectives
        Index_JJ = list ( np.where( tagg[1] == "JJ"))
        Index_JJR = list ( np.where( tagg[1] == "JJR"))
        Index_JJS = list ( np.where( tagg[1] == "JJS"))
            
            #adverbs
        Index_RB = list ( np.where( tagg[1] == "RB"))
        Index_RBR = list ( np.where( tagg[1] == "RBR"))
        Index_RBS = list ( np.where( tagg[1] == "RBS"))
            
            #IN	preposition/subordinating conjunction
        Index_IN = list ( np.where( tagg[1] == "IN"))

            #CC	coordinating conjunction
        Index_CC = list ( np.where( tagg[1] == "CC"))

        Index_T= len(Index_VB)+len(Index_VBD)+len(Index_VBG)+len(Index_VBN)+ len(Index_VBP)+ len(Index_VBZ)+len(Index_JJ )+len(Index_JJR )+len(Index_JJS )+len(Index_RB)+len(Index_RBR)+len(Index_RBS)+len(Index_IN)+len(Index_CC)

        Density_P=np.append(Density_P, (Index_T/len(word_tokens)/len(Sent_doc)))
        #Co_5=sum(Density_P)/len(Sent_doc)
        #Coh_M=np.append(  Coh_M,  Co_5)
    return[Density_P]             
##
##2. The Content Density
##..........................................................................        
#
def Coherence_S_8(docs_A): 
    #Measure the content density . 
    Density_C=np.array([])
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        Sent_doc=sent_tokenize(doc)
        N_N=0
        N_P=0
        ii=0
        
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(Sent_doc[ii])
        word_tokens = [w for w in tokenized_word if w.isalpha()]
        tagg =[]
        Index_NNP =[]
        Index_PRP =[]
        tagg=pd.DataFrame(nltk.pos_tag(word_tokens))
        # NN	noun
        Index_NN = list  ( np.where( tagg[1] == "NN" ))  ####some error here
        Index_NNP = list ( np.where( tagg[1] == "NNP" ))  ####some error here
        Index_NNPS = list ( np.where( tagg[1] == "NNPS" ))  ####some error here
            
            #verb

        Index_VB = list  ( np.where( tagg[1] == "VB" ))  ####some error here
        Index_VBD = list ( np.where( tagg[1] == "VBD" ))  ####some error here
        Index_VBG = list ( np.where( tagg[1] == "VBG" ))  ####some error here
        Index_VBN = list ( np.where( tagg[1] == "VBN" ))  ####some error here
        Index_VBP = list ( np.where( tagg[1] == "VBP" ))  ####some error here
        Index_VBZ = list ( np.where( tagg[1] == "VBZ" ))  ####some error here
                
            #adjectives
        Index_JJ = list ( np.where( tagg[1] == "JJ"))
        Index_JJR = list ( np.where( tagg[1] == "JJR"))
        Index_JJS = list ( np.where( tagg[1] == "JJS"))
            
            #adverbs
        Index_RB = list ( np.where( tagg[1] == "RB"))
        Index_RBR = list ( np.where( tagg[1] == "RBR"))
        Index_RBS = list ( np.where( tagg[1] == "RBS"))                            
#            
        Index_T= len(Index_VB)+len(Index_VBD)+len(Index_VBG)+len(Index_VBN)+ len(Index_VBP)+ len(Index_VBZ)+len(Index_JJ )+len(Index_JJR )+len(Index_JJS )+len(Index_RB)+len(Index_RBR)+len(Index_RBS)+len(Index_NNP)+len(Index_NN)+len(Index_NNPS)
        Density_C=np.append(Density_C, Index_T/(len(word_tokens)*len(Sent_doc)))
            
        Coh_M=Density_C
    return[Coh_M]         


   
No_Measure=8
No_Subject=14

h=np.zeros((No_Measure, No_Subject))
    

h[0][:]=np.asarray(Coherence_S_1( docs_A[0]))
h[1][:]=np.asarray(Coherence_S_2( docs_A[0]))
h[2][:]=np.asarray(Coherence_S_3( docs_A[0]))
h[3][:]=np.asarray(Coherence_S_4( docs_A[0]))
h[4][:]=np.asarray(Coherence_S_5( docs_A[0]))
h[5][:]=np.asarray(Coherence_S_6( docs_A[0]))
h[6][:]=np.asarray(Coherence_S_7( docs_A[0]))
h[7][:]=np.asarray(Coherence_S_8( docs_A[0]))

hh=np.transpose(h)
class_names=['Depression','Psychosis' ]
#dataset = pd.DataFrame({'Lexical_Diversity':hh[:,0],'Brunet_Index':hh[:,1],'Honore_Satistic':hh[:,2],'Flesch Reading':hh[:,3], 'Flesch-Kincaid':hh[:,4], 'Ambigous_Pronouns':hh[:,5],'First_Pronouns_Ratio':hh[:,6], 'Third_Pronouns_Ratio':hh[:,7], 'Noun_Verb_Ratio':hh[:,8], 'Noun_Ratio':hh[:,9],'Subordinate_Coordinate_Ratio':hh[:,12],'Propositional_Density':hh[:,11], 'Content_Density':hh[:,12],  'Tangentiality':hh[:,13], 'Incoherence_SA_1':hh[:,14],'Incoherence_SA_2':hh[:,15],'Incoherence_SA_3':hh[:,16] })                     
dataset = pd.DataFrame({'Ambigous_Pronouns':hh[:,0],'First_Pronouns_Ratio':hh[:,1], 'Third_Pronouns_Ratio':hh[:,2], 'Noun_Verb_Ratio':hh[:,3], 'Noun_Ratio':hh[:,4],'Subordinate_Coordinate_Ratio':hh[:,5],'Propositional_Density':hh[:,6], 'Content_Density':hh[:,7] })

                 
SZ_Type = ['Depression','Depression', 'Depression','Depression','Depression','Depression', 'Depression','Psychosis', 'Psychosis','Psychosis', 'Psychosis','Psychosis','Psychosis','Psychosis']                      
dataset['SZ_Type']=SZ_Type
                    
                 

