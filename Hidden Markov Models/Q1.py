#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 12:56:04 2017

@author: rakadalal
"""

import json



def pos_vocabulary(filename):
    word_list_list=[]
    pos_list_list=[]
    with open(filename) as data_file:    
        data = json.load(data_file)
    for item in data:
        word_list=[]
        pos_list=[]
        for i in range(0,len(data[item])):
            pos=data[item][i]['pos']
            pos_list.append(pos)
            word=data[item][i]['text']
            word_list.append(word)
        word_list_list.append(word_list)
        pos_list_list.append(pos_list)
    return (word_list_list, pos_list_list)

def count_pos(pos_list_list):
    pos=[]
    for i in range(0,len(pos_list_list)):
        for item in pos_list_list[i]:
            pos.append(item)
    pos=set(pos)
    
    return len(pos)
    
def common_tuples(word_list_list):
    common_tuples_dict={}
    for i in range(0,len(word_list_list)):
        for j in range(0,len(word_list_list[i])):
            if word_list_list[i][j]=="to":
                if j>0 and j<(len(word_list_list[i])-1):
                    tup=(word_list_list[i][j-1],word_list_list[i][j],word_list_list[i][j+1])
                    if tup in common_tuples_dict:
                        common_tuples_dict[tup]+=1
                    else:
                        common_tuples_dict[tup]=1   
    popular_tuples = sorted(common_tuples_dict, key = common_tuples_dict.get, reverse = True)
    top_10 = popular_tuples[:10]
    return top_10

def all_ways_to(word_list_list, pos_list_list):
    pos_to=[]
    for i in range(0,len(word_list_list)):
        for j in range(0,len(word_list_list[i])):
            if word_list_list[i][j]=="to":
                pos_to.append(pos_list_list[i][j])
    pos_to=set(pos_to)
    return pos_to
            
def syntactic_patterns(word_list_list, pos_list_list):
    common_tuples_dict={}
    for i in range(0,len(word_list_list)):
        for j in range(0,len(word_list_list[i])):
            if word_list_list[i][j]=="to":
                if j>0 and j<(len(word_list_list[i])-1):
                    tup=(pos_list_list[i][j-1],pos_list_list[i][j],pos_list_list[i][j+1])
                    if tup in common_tuples_dict:
                        common_tuples_dict[tup]+=1
                    else:
                        common_tuples_dict[tup]=1   
    popular_tuples = sorted(common_tuples_dict, key = common_tuples_dict.get, reverse = True)
    top_20 = popular_tuples[:20]
    return top_20    

def RP_X(word_list_list, pos_list_list):
    flag=0
    for i in range(0,len(pos_list_list)):
        for j in range(0,len(pos_list_list[i])):
            if pos_list_list[i][j]=="RP":
                word=word_list_list[i][j]
                first=word_list_list[i]
                flag=1
                break
        if flag==1:
            break
    flag=0   
    for i in range(0,len(word_list_list)):
        for j in range(0,len(word_list_list[i])):
            if word_list_list[i][j]==word:
                if pos_list_list[i][j]!="RP":
                    X=pos_list_list[i][j]
                    second=word_list_list[i]
                    flag=1
                    break
        if flag==1:
            break
    sentence1=""
    sentence2=""
    for item in first:
        sentence1+=str(item)+" " 
    for item in second:
        sentence2+=str(item)+" "                 
    
    return (word, X, sentence1, sentence2)
    
word_list_list, pos_list_list=pos_vocabulary('data/train.json')
tags=count_pos(pos_list_list)
print ("Number of different part of speech tag types defined: "+str(tags))
print ("\n")
top_10=common_tuples(word_list_list)
print ("The ten most common tuples centered around the word “to”:")
for item in top_10:
    print item
print ("\n")
pos=all_ways_to(word_list_list, pos_list_list)
print ("'to' is defined as:")
for item in pos:
    print item
print ("\n")
top_20=syntactic_patterns(word_list_list, pos_list_list)
print ("The twenty most common syntactic patterns surrounding the word “to”:")
for item in top_20:
    print item
word, X, sentence1, sentence2=RP_X(word_list_list, pos_list_list)
print ("The word is "+str(word))
print ("The other tag is "+str(X)+"\n")
print ("The sentence when "+str(word)+" is used as RP:\n"+str(sentence1)+"\n")
print ("The sentence when "+str(word)+" is used as "+str(X)+":\n"+str(sentence2)+"\n")