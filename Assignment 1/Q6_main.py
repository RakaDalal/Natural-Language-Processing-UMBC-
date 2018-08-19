#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:51:52 2017

@author: rakadalal
"""
from Q6 import *
import pickle
import gzip
import sys


def main():
    #input variables
    model_name=sys.argv[1]
    N=int(sys.argv[2])
    training=sys.argv[3]
    development=sys.argv[4]
    path_to_model=sys.argv[5]
    
    #creating unigram, bigram, trigram dictionary from training file
    oov = (least_common_words(vocabCreate(training)))
    unigram_dict=unigramscount(training,oov)
    bigram_dict=bigramscount(training,oov)
    trigram_dict=trigramscount(training,oov)
    
    #storing the dictionaries
    f = gzip.open(path_to_model+'unigram_dict.pklz','wb')
    pickle.dump(unigram_dict,f)
    f.close()

    f = gzip.open(path_to_model+'bigram_dict.pklz','wb')
    pickle.dump(bigram_dict,f)
    f.close()

    f = gzip.open(path_to_model+'trigram_dict.pklz','wb')
    pickle.dump(trigram_dict,f)
    f.close()

    #counting from development file
    (unigram_list,bigram_list,trigram_list,unigram_set,bigram_set,trigram_set)=counting(development)
    
    model_name=model_name.lower()
    
    #calling the method based on model_name and N
    if N==0:
        print ("The perplexity is: "+str(zerogram(unigram_dict)))
        param=0
    else:
        if model_name=="laplace":
            param=evaluation_laplace(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,N)
        else:
            if model_name=="interpolation":
                if N==1:
                    param=evaluation_interpolate_unigram(unigram_list,unigram_dict)
                elif N==2:
                    param=evaluation_interpolate_bigram(unigram_list,bigram_list,unigram_dict,bigram_dict)
                else:
                    param=evaluation_interpolate_trigram(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict)
            else:
                if N==1:
                    param=evaluation_backoff_unigram(unigram_list,unigram_dict,unigram_set)
                elif N==2:
                    param=evaluation_backoff_bigram(unigram_list,bigram_list,unigram_dict,bigram_dict,unigram_set,bigram_set)
                else:
                    param=evaluation_backoff_trigram(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,unigram_set,bigram_set,trigram_set)
    model={}
    model["name"]= model_name
    model["N"]=N
    model["param"]=param
         
    #storing the serialized model
    f = gzip.open(path_to_model+'model.pklz','wb')
    pickle.dump(model,f)
    f.close()

            

if __name__=='__main__' :
    main()