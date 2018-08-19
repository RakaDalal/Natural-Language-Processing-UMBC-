#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:09:02 2017

@author: rakadalal
"""

from Q6 import *
import pickle
import gzip
import sys

def main():
    #input variables
    path_to_model=sys.argv[1]
    test=sys.argv[2]
    
    #reading dictionaries from files
    f = gzip.open(path_to_model+'unigram_dict.pklz','rb')
    unigram_dict = pickle.load(f)
    f.close()
    f = gzip.open(path_to_model+'bigram_dict.pklz','rb')
    bigram_dict = pickle.load(f)
    f.close()
    f = gzip.open(path_to_model+'trigram_dict.pklz','rb')
    trigram_dict = pickle.load(f)
    f.close()
    f = gzip.open(path_to_model+'model.pklz','rb')
    model = pickle.load(f)
    f.close()
    
    #counting from test file
    (unigram_list,bigram_list,trigram_list,unigram_set,bigram_set,trigram_set)=counting(test)
    model_name=model["name"].lower()
    N=model["N"]
    param=model["param"]
    
    #calling appropriate method with same setof parameters
    if int(N)==0:
        perplexity = (zerogram(unigram_dict))
    else:
        if model_name=="laplace":
            if int(N)==1:
                perplexity=laplace_unigram(unigram_list,unigram_dict,float(param))
            elif int(N)==2:
                perplexity=laplace_bigram(unigram_list,bigram_list,unigram_dict,bigram_dict,float(param))
            else:
                perplexity=laplace_trigram(bigram_list,trigram_list,bigram_dict,trigram_dict,float(param))
        else:
            if model_name=="interpolation":
                if int(N)==1:
                    perplexity=interpolate_unigram(unigram_list, unigram_dict,float(param))
                elif int(N)==2:
                    perplexity=interpolate_bigram(unigram_list, bigram_list, unigram_dict,bigram_dict,float(param[0]),float(param[1]))
                else:
                    perplexity=interpolate_trigram(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,float(param[0]),float(param[1]),float(param[2]))
            else:
                if int(N)==1:
                    perplexity=backoff_unigram(unigram_list,unigram_dict,unigram_set,float(param[0]),float(param[1]))
                elif int(N)==2:
                    perplexity=backoff_bigram(unigram_list,bigram_list,unigram_dict,bigram_dict,unigram_set,bigram_set,float(param[0]),float(param[1]),float(param[2]))
                else:
                    perplexity=backoff_trigram(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,unigram_set,bigram_set,trigram_set,float(param[0]),float(param[1]),float(param[2]),float(param[3]))
    print ("The perplexity is: "+str(perplexity))
            

if __name__=='__main__' :
    main()