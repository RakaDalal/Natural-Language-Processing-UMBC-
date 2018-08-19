#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:06:53 2017

@author: rakadalal
"""

import codecs
import operator
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
import math
import random


def countlines(filename):
    filepointer=codecs.open(filename,'r',encoding='utf-8')
    filep=filepointer.readlines()
    count=0
    for line in filep:
        count+=1
    return count
    
def word_types_and_tokens(filename):
    filepointer=codecs.open(filename,'r',encoding='utf-8')
    filep=filepointer.readlines()
    wordType={}
    for line in filep:
        line=line.split(" ")
        for item in line:
            if item.strip() in wordType:
                wordType[(item.strip())]+=1
            else:
                wordType[(item.strip())]=1      
    tup=(wordType,sum(wordType.values()),len(wordType))
    return tup

def most_common_words(wordType):
    count=0
    for key, value in sorted(wordType.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        print "%s: %s" % (key, value)
        count+=1
        if count==100:
            break
        
def least_common_words(wordType):
    count=0
    for key, value in sorted(wordType.iteritems(), key=lambda (k,v): (v,k)):
        print "%s: %s" % (key, value)
        count+=1
        if count==100:
            break
        
def out_of_vocabulary(trainingDict, developmentDict):
    oov={}
    for key in developmentDict:
        if key not in trainingDict:
            oov[key]=developmentDict[key]
    return oov

def group_by_value(dictionary):
    x=[]
    y=[]
    v = defaultdict(list)
    for key, value in sorted(dictionary.iteritems(), key=lambda (k,v): (v,k)):
        v[dictionary[key]].append(key)
    for i in range (1,(max(v.keys())+1)):
        if i in v:
            y.append(len(v[i]))
        else:
            y.append(0)
        x.append(i)
    tup = (x,y)
    return tup
    
def plot(X,Y,x_label,y_label):
    plt.plot(X, Y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.show()
    
    
def zipf_law(filename):
    text=[]
    originalFC={}
    lowerFC={}
    sampledFC={}
    sampledLowerFC={}
    totalLines=countlines(filename)
    sampleValue=totalLines/2
    randomNum=[]
    randomText=[]
    filepointer=codecs.open(filename,'r',encoding='utf-8')
    filep=filepointer.readlines()
    for line in filep:
        text.append(line)
    for line in text:
        try:
            line=str(line).split(" ")
            for item in line:
                if item.strip() in originalFC:
                    originalFC[(item.strip())]+=1
                else:
                    originalFC[(item.strip())]=1
                if (item.strip()).lower() in lowerFC:
                    lowerFC[(item.strip()).lower()]+=1
                else:
                    lowerFC[(item.strip()).lower()]=1
            
        except:
            continue
    
    for x in range(sampleValue):
        randomNum.append(random.randint(0,(totalLines-1))) 
    for i in randomNum:
        randomText.append(text[i])
    for line in randomText:
        try:
            line=str(line).split(" ")
            for item in line:
                if item.strip() in sampledFC:
                    sampledFC[(item.strip())]+=1
                else:
                    sampledFC[(item.strip())]=1
                if (item.strip()).lower() in sampledLowerFC:
                    sampledLowerFC[(item.strip()).lower()]+=1
                else:
                    sampledLowerFC[(item.strip()).lower()]=1
        except:
            continue

    tup=(originalFC,lowerFC,sampledFC,sampledLowerFC)
    return (tup)
    
    
    
    
    
def rank_and_plot(wordfreq):
    f=[]
    r=[]
    for key, value in sorted(wordfreq.iteritems(), key=lambda (k,v): (v,k), reverse=True):
        f.append(math.log(value))
    maximum=len(f)
    for i in range (1, maximum+1):
        r.append(math.log(i))
    plot(f,r,"log frequency", "log rank")
    slope, intercept, r_value, p_value, std_err = stats.linregress(f,r)
    print (slope)
    print (r_value*r_value)
