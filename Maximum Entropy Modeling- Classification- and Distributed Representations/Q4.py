#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 12:41:55 2017

@author: rakadalal
"""

import gzip
import numpy
import math
import string
import csv

#This method creates the outcome vocabulary
def outcome_vocab(filename):
    with gzip.open(filename, 'rb') as filepointer:
        filep=filepointer.readlines()
    vocabulary={}
    for line in filep:
        line=line.strip("\n")
        for item in line:
            if item in vocabulary:
                vocabulary[item]+=1
            else:
                vocabulary[item]=1
    vocabulary['E']=1
    vocabulary['U']=0
    return vocabulary

#This method creates the context vocabulary
def context_vocab(filename):
    with gzip.open(filename, 'rb') as filepointer:
        filep=filepointer.readlines()
    vocabulary={}
    for line in filep:
        line=line.strip("\n")
        for item in line:
            if item in vocabulary:
                vocabulary[item]+=1
            else:
                vocabulary[item]=1
    vocabulary['B']=1
    vocabulary['U']=0
    return vocabulary

#This method returns the theta needed in part (vii) for unigram maxent model.
def theta(feature):
    return (numpy.tanh((ord(feature) - 74)/128.0))

#This method returns the theta needed in part (vii) for bigram maxent model.
def bigram_theta(feature1,feature2):
    return (numpy.tanh((ord(feature2) - ord(feature1))/128.0))

#This model produces the eight column for unigram model
def eight_columns_unigram(outcome_vocabulary):
    first=[]
    second=[]
    third=[]
    fourth=[]
    fifth=[]
    sixth=[]
    seventh=[]
    eighth=[]
    h=0.000001
    N=sum(outcome_vocabulary.values())
    V=len(outcome_vocabulary)
    total=0
    for item in outcome_vocabulary:
        total+=math.exp(theta(item))    
    for item in outcome_vocabulary:
        if item==" ":
            first.append("space")
        else:
            first.append(item)
        second.append(outcome_vocabulary[item])
        third.append(N/float(V))
        count=0
        for item2 in outcome_vocabulary:
            if item2 == item:
                continue
            count+=(outcome_vocabulary[item2]*math.log(1/(math.exp(h)+(V-1))))
        fourth.append((((outcome_vocabulary[item]*math.log(math.exp(h)/(math.exp(h)+(V-1)))+count)/float(N))-math.log(1/float(V)))/float(h))
        fifth.append(N/float(V))
        count=0
        for item2 in outcome_vocabulary:
            if item2 == item:
                continue
            count+=(outcome_vocabulary[item2]*math.log(math.exp(1)/(math.exp(h+1)+((V-1)*math.exp(1)))))
        sixth.append((((outcome_vocabulary[item]*math.log(math.exp(h+1)/(math.exp(h+1)+((V-1)*math.exp(1))))+count)/float(N))-math.log(1/float(V)))/float(h))
        seventh.append((math.exp(theta(item))*N)/float(total))
        total_theta=0
        for item2 in outcome_vocabulary:
            if item2 == item:
                continue
            total_theta+=math.exp(theta(item))
        count=0
        original=0
        for item2 in outcome_vocabulary:
            if item2 == item:
                continue
            count+=(outcome_vocabulary[item2]*math.log(math.exp(theta(item2))/(math.exp(h+theta(item))+total_theta)))
            original+=(outcome_vocabulary[item2]*math.log(math.exp(theta(item2))/(math.exp(theta(item))+total_theta)))
#            print (((outcome_vocabulary[item]*math.log(math.exp(h+theta(item))/(math.exp(h+theta(item))+total_theta)))+count)/float(N))-((math.log((outcome_vocabulary[item]*math.log(math.exp(theta(item2))/(math.exp(theta(item))+float(total_theta)+original))))))
        eighth.append(((((outcome_vocabulary[item]*math.log(math.exp(h+theta(item))/(math.exp(h+theta(item))+total_theta)))+count)/float(N))-(((outcome_vocabulary[item]*math.log(math.exp(theta(item))/(math.exp(theta(item))+total_theta)))+original)/float(N)))/float(h))
        
    
    return (first,second,third,fourth,fifth,sixth,seventh,eighth)
       
    
#This method creates the bigram vocabulary    
def bigram_vocab(outcome_vocabulary,context_vocabulary,filename):
    bigram_vocabulary={}
    with gzip.open(filename, 'rb') as filepointer:
        filep=filepointer.readlines()
        for line in filep:
            line=line.strip("\n")
            for i in range(0,len(line)-1):
                feature=line[i]+line[i+1]
                if feature in bigram_vocabulary:
                    bigram_vocabulary[feature]+=1
                else:
                    bigram_vocabulary[feature]=1
    for i in context_vocabulary:
        for j in outcome_vocabulary:
            feature=i+j
            if feature in bigram_vocabulary:
                continue
            else:
                if "B" in feature and line[0]==feature[1]:
                    bigram_vocabulary[feature]=1
                elif "E" in feature and line[len(line)-1]==feature[0]:
                    bigram_vocabulary[feature]=1
                else:
                    bigram_vocabulary[feature]=0

    return (bigram_vocabulary)
   
#This model produces the eight column for bigram model
def eight_columns_bigram(bigram_vocabulary,outcome_vocabulary):
    #creating the feature vector
    feature=[]
    for item in bigram_vocabulary:
        feature.append(item)
    for item in outcome_vocabulary:
        feature.append(item)
        
    first=[]
    second=[]
    third=[]
    fourth=[]
    fifth=[]
    sixth=[]
    seventh=[]
    eighth=[]
    h=0.000001
    N=sum(bigram_vocabulary.values())
    V=len(bigram_vocabulary)
    unigram_V=len(outcome_vocabulary)
    total=0
    for item in bigram_vocabulary:
        value1=bigram_theta(item[0],item[1])
        value2=theta(item[1])
        total+=math.exp(value1+value2)  
                 
    for item in feature:
        if " " in item:
            item2=string.replace(item," ","space")
            first.append(item2)
        else:
            first.append(item)
        if item in bigram_vocabulary:
            second.append(bigram_vocabulary[item])
        else:
            count=0
            for bigram in bigram_vocabulary:
                if bigram[1]==item:
                    count+=bigram_vocabulary[bigram]
            second.append(count)
        if item in bigram_vocabulary:
            third.append(N/float(V))
        else:
            third.append((N*unigram_V)/float(V))
        if item in bigram_vocabulary:
            count=0
            for item2 in bigram_vocabulary:
                if item2 == item:
                    continue
                count+=(bigram_vocabulary[item2]*math.log(1/(math.exp(h)+(V-1))))
            fourth.append((((bigram_vocabulary[item]*math.log(math.exp(h)/(math.exp(h)+(V-1)))+count)/float(N))-math.log(1/float(V)))/float(h))
        else:
            denom=0
            for bigram in bigram_vocabulary:
                if bigram[1]==item:
                    denom+=math.exp(h)
                else:
                    denom+=1
            count=0
            for bigram in bigram_vocabulary:
                if bigram[1]==item:
                    count+=(bigram_vocabulary[bigram]*math.log(math.exp(h)/denom))
                else:
                    count+=(bigram_vocabulary[bigram]*math.log(1/denom))
            fourth.append((((count)/float(N))-math.log(1/float(V)))/float(h))
        
        if item in bigram_vocabulary:
            fifth.append(N/float(V))
        else:
            fifth.append((N*unigram_V)/float(V))
            
        if item in bigram_vocabulary:
            count=0
            for item2 in bigram_vocabulary:
                if item2 == item:
                    continue
                count+=(bigram_vocabulary[item2]*math.log((math.exp(2))/(math.exp(h+2)+((V-1)*math.exp(2)))))
            sixth.append((((bigram_vocabulary[item]*math.log(math.exp(h+2)/(math.exp(h+2)+((V-1)*math.exp(2))))+count)/float(N))-math.log(1/float(V)))/float(h))
        else:
            denom=0
            for bigram in bigram_vocabulary:
                if bigram[1]==item:
                    denom+=math.exp(h+2)
                else:
                    denom+=math.exp(2)
            count=0
            for bigram in bigram_vocabulary:
                if bigram[1]==item:
                    count+=(bigram_vocabulary[bigram]*math.log(math.exp(2+h)/denom))
                else:
                    count+=(bigram_vocabulary[bigram]*math.log(math.exp(2)/denom))
            sixth.append((((count)/float(N))-math.log(1/float(V)))/float(h))
            
        if item in bigram_vocabulary:
            value1=bigram_theta(item[0],item[1])
            value2=theta(item[1])
            seventh.append((N*math.exp(value1+value2))/float(total))
        else:
            num_total=0
            for bigram in bigram_vocabulary:
                if bigram[1]==item:
                    value1=bigram_theta(bigram[0],bigram[1])
                    value2=theta(item)
                    num_total+=math.exp(value1+value2) 
            seventh.append((N*num_total)/float(total))
            
        if item in bigram_vocabulary:
            total_theta=0
            for item2 in bigram_vocabulary:
                if item2 == item:
                    continue
                value1=bigram_theta(item[0],item[1])
                value2=theta(item[1])
                total_theta+=math.exp(value1+value2)
            count=0
            original=0
            for item2 in bigram_vocabulary:
                if item2 == item:
                    continue
                value1=bigram_theta(item[0],item[1])
                value2=theta(item[1])
                count+=(bigram_vocabulary[item2]*math.log((math.exp(value1+value2))/(total_theta+math.exp(value1+value2+h))))
                original+=(bigram_vocabulary[item2]*math.log((math.exp(value1+value2))/(total_theta+math.exp(value1+value2))))
            value1=bigram_theta(item[0],item[1])
            value2=theta(item[1])
            eighth.append(((((bigram_vocabulary[item]*math.log(math.exp(h+value1+value2)/(math.exp(h+value1+value2)+total_theta)))+count)/float(N))-(((bigram_vocabulary[item]*math.log(math.exp(value1+value2)/(math.exp(value1+value2)+total_theta)))+original)/float(N)))/float(h))
        else:
            denom=0
            original_denom=0
            for bigram in bigram_vocabulary:
                value1=bigram_theta(bigram[0],bigram[1])
                value2=theta(item)
                if bigram[1]==item:
                    denom+=math.exp(value1+value2+h)
                else:
                    denom+=math.exp(value1+value2)
                original_denom+=math.exp(value1+value2)
            count=0
            original=0
            for bigram in bigram_vocabulary:
                value1=bigram_theta(bigram[0],bigram[1])
                value2=theta(item)
                if bigram[1]==item:
                    count+=(bigram_vocabulary[bigram]*math.log(math.exp(value1+value2+h)/denom))
                else:
                    count+=(bigram_vocabulary[bigram]*math.log(math.exp(value1+value2)/denom))
                original+=(bigram_vocabulary[bigram]*math.log(math.exp(value1+value2)/original_denom))
            eighth.append((((count)/float(N))-((original)/float(N)))/float(h))


    return (first,second,third,fourth,fifth,sixth,seventh,eighth)

#This method is for writing in file
def writing_in_csv(filename,first,second,third,fourth,fifth,sixth,seventh,eighth ):
    filep = open(filename, 'w')
    writer = csv.writer(filep)
    new_row = ["First_column","Second_column","Third_column","Fourth_column","Fifth_column","Sixth_column","Seventh_column","Eighth_column"]
    writer.writerow(new_row)
    for i in range(0,len(first)):
        new_row = [first[i], second[i], third[i], fourth[i], fifth[i], sixth[i], seventh[i], eighth[i]]
        writer.writerow(new_row)
   
   

    
outcome_vocabulary=outcome_vocab("train.5k.processed.txt.gz")
context_vocabulary=context_vocab("train.5k.processed.txt.gz")
bigram_vocabulary=bigram_vocab(outcome_vocabulary,context_vocabulary,"train.5k.processed.txt.gz")
(first,second,third,fourth,fifth,sixth,seventh,eighth)=eight_columns_unigram(outcome_vocabulary)
writing_in_csv("unigram_maxent.csv",first,second,third,fourth,fifth,sixth,seventh,eighth )
(first,second,third,fourth,fifth,sixth,seventh,eighth)=eight_columns_bigram(bigram_vocabulary,outcome_vocabulary)
writing_in_csv("bigram_maxent.csv",first,second,third,fourth,fifth,sixth,seventh,eighth )

