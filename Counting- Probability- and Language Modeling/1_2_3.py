#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:18:02 2017

@author: rakadalal
"""
from methods import *


def main():
    count=countlines("train.5k.txt")
    print ("Total number of sentences in training split: "+ str(count))
    print ("Average number of sentences in each training document: "+ str(count/5000))
    count=countlines("dev.2k.txt")
    print ("Total number of sentences in development split: "+ str(count))
    print ("Average number of sentences in each development document: "+ str(count/2000))
    (dic_train,tokens,wordTypes)=word_types_and_tokens("train.5k.txt")
    print ("\nTotal number of tokens in training split:"+str(tokens))
    print ("Total number of different word types in training split:"+str(wordTypes))
    print ("\nMost common words:")
    most_common_words(dic_train)
    print ("\nLeast common words:")
    least_common_words(dic_train)
    (dic_dev,tokens,wordTypes)=word_types_and_tokens("dev.2k.txt")
    print ("\nTotal number of tokens in development split:"+str(tokens))
    print ("Total number of different word types in development split:"+str(wordTypes))
    oov=out_of_vocabulary(dic_train, dic_dev)
    print ("\nNumber of out of vocabulary words:"+str(len(oov)))
    print("\nPlot for training split")
    (X,Y)=group_by_value(dic_train)
    plot(X,Y,"n","|V(n)|")
    print ("\nPlot for out of vocabulary words")
    (X,Y)=group_by_value(oov)
    plot(X,Y,"n","|U(n)|")
    (originalFC,lowerFC,sampledFC,sampledLowerFC)=zipf_law("train.5k.txt")
    rank_and_plot(originalFC)
    rank_and_plot(lowerFC)
    rank_and_plot(sampledFC)
    rank_and_plot(sampledLowerFC)

if __name__=='__main__' :
    main()