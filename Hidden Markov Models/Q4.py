#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 19:11:57 2017

@author: rakadalal
"""

import json
import math
import operator

def pos_vocabulary(filename):
    word_list_list=[]
    pos_list_list=[]
    tags=[]
    tokens=[]
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
            tags.append(pos)
            tokens.append(word)
        word_list_list.append(word_list)
        pos_list_list.append(pos_list)
    tags=set(tags)
    tokens=set(tokens)
    return (word_list_list, pos_list_list, tags, tokens)


def counts_from_training(word_list_list, pos_list_list, tags):
    emission_dict={}
    transition_dict={}
    for item in tags:
        emission_dict[item]={}
        transition_dict[item]={}
    transition_dict["BOS"]={}
    for i in range(0,len(pos_list_list)):
        for j in range(0,len(pos_list_list[i])):
            if word_list_list[i][j] in emission_dict[pos_list_list[i][j]]:
                emission_dict[pos_list_list[i][j]][word_list_list[i][j]]+=1
            else:
                emission_dict[pos_list_list[i][j]][word_list_list[i][j]]=1
    for i in range(0,len(pos_list_list)):
        previous="BOS"
        for j in range(0,len(pos_list_list[i])):
            if pos_list_list[i][j] in transition_dict[previous]:
                transition_dict[previous][pos_list_list[i][j]]+=1
            else:
                transition_dict[previous][pos_list_list[i][j]]=1
            previous=pos_list_list[i][j]
    
    return (emission_dict, transition_dict)

def development(filename):
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
            
def model(word_list_list, pos_list_list, emission_dict, transition_dict, tokens, tags, alpha):
    J=0
    for i in range(0,len(word_list_list)):
        previous="BOS"
        prob=1
        for j in range(0,len(word_list_list[i])):
            if word_list_list[i][j] in emission_dict[pos_list_list[i][j]]:
                upper=emission_dict[pos_list_list[i][j]][word_list_list[i][j]]+alpha
            else:
                upper=alpha
            lower=sum(emission_dict[pos_list_list[i][j]].values())+(alpha*len(tokens))
            emission_prob=upper/float(lower)
            if pos_list_list[i][j] in transition_dict[previous]:
                upper=transition_dict[previous][pos_list_list[i][j]]+alpha
            else:
                upper=alpha
            lower=sum(transition_dict[previous].values())+(alpha*len(tags))
            transition_prob=upper/float(lower)
            if transition_prob>=1:
                print transition_prob
            previous=pos_list_list[i][j]
            prob=prob*emission_prob*transition_prob
        J+=math.log(prob)
    J=J/float(len(word_list_list))
    return J
    
def tuning(word_list_list, pos_list_list, emission_dict, transition_dict, tokens, tags):
    max_value=-999
    i=0.01
    while(i<0.1):
        J=model(word_list_list, pos_list_list, emission_dict, transition_dict, tokens, tags, i)
        if J>max_value:
            max_value=J
            alpha=i
        i+=0.01
    return alpha
        
def viterbi(word_list_list, tokens, tags, emission_dict, transition_dict, alpha):
    predicted_list_list=[]
    
    for i in range(0,len(word_list_list)):
        v={}
        b={}
        predicted_list=[]
        previous="BOS"
        for j in range(0,len(word_list_list[i])):
            v[j]={}
            b[j]={}
            for t in tags:
                if word_list_list[i][j] in emission_dict[t]:
                    upper=emission_dict[t][word_list_list[i][j]]+alpha
                else:
                    upper=alpha
                lower=sum(emission_dict[t].values())+(alpha*len(tokens))
                emission_prob=upper/float(lower)
                emission_prob=math.log(emission_prob)
                if j==0:
                    if t in transition_dict[previous]:
                        upper=transition_dict[previous][t]+alpha
                    else:
                        upper=alpha
                    lower=sum(transition_dict[previous].values())+(alpha*len(tags))
                    transition_prob=upper/float(lower)
                    transition_prob=math.log(transition_prob)
                    v[j][t]=emission_prob+transition_prob
                    b[j][t]=previous
                else:
                    v[j][t]=-9999
                    for old in tags:
                        if t in transition_dict[old]:
                            upper=transition_dict[old][t]+alpha
                        else:
                            upper=alpha
                        lower=sum(transition_dict[old].values())+(alpha*len(tags))
                        transition_prob=upper/float(lower)
                        transition_prob=math.log(transition_prob)
                        if ((v[j-1][old]+emission_prob+transition_prob)>v[j][t]):
                            v[j][t]=v[j-1][old]+emission_prob+transition_prob
                            b[j][t]=old
        k=len(word_list_list[i])-1   
        pos_tag=max(v[len(word_list_list[i])-1].iteritems(), key=operator.itemgetter(1))[0]
        predicted_list.append(pos_tag)
        while (k>0):
            pos_tag=b[k][pos_tag]
            predicted_list.append(pos_tag)
            k-=1
        predicted_list.reverse()
        predicted_list_list.append(predicted_list)
    return predicted_list_list    
            
def evaluation(true_pos_list_list, predicted_pos_list_list):
    correct={}
    incorrect={}
    for t in tags:
        correct[t]=0
        incorrect[t]=0
    
    per_token_accuracy=0
    per_sentence_accuracy=0
    total_tokens=0
    for i in range(0,len(predicted_pos_list_list)):
        flag="True"
        for j in range(0,len(predicted_pos_list_list[i])):
            if predicted_pos_list_list[i][j]==true_pos_list_list[i][j]:
                correct[true_pos_list_list[i][j]]+=1
                per_token_accuracy+=1
            else:
                incorrect[true_pos_list_list[i][j]]+=1
                flag="False"
            total_tokens+=1
        if flag=="True":
            per_sentence_accuracy+=1
    per_token_accuracy=per_token_accuracy/float(total_tokens)
    per_sentence_accuracy=per_sentence_accuracy/float(len(predicted_pos_list_list))
    print ("Per token accuracy:"+str(per_token_accuracy))
    print ("Per sentence accuracy:"+str(per_sentence_accuracy))
    for t in tags:
        s=correct[t]/float(correct[t]+incorrect[t])
        incorrect[t]=incorrect[t]/float(correct[t]+incorrect[t])
        correct[t]=s
               
    sorted_correct = sorted(correct, key = correct.get, reverse = True)
    top_5 = sorted_correct[:5]
    print ("The top 5 POS tags that are most commonly identified correctly:")
    for item in top_5:
        print item
    sorted_incorrect = sorted(incorrect, key = incorrect.get, reverse = True)
    top_5 = sorted_incorrect[:5]
    print ("The top 5 POS tags that are most commonly identified incorrectly:")
    for item in top_5:
        print item
            
    
        
                
                
                
            
            
            
    
    

word_list_list, pos_list_list, tags, tokens=pos_vocabulary('data/train.json')
emission_dict, transition_dict=counts_from_training(word_list_list, pos_list_list, tags)
word_list_list, pos_list_list=development('data/dev.json')
alpha=tuning(word_list_list, pos_list_list, emission_dict, transition_dict, tokens, tags)
print alpha
word_list_list, true_pos_list_list=development('data/test.json')
predicted_pos_list_list=viterbi(word_list_list, tokens, tags, emission_dict, transition_dict, alpha)
evaluation(true_pos_list_list, predicted_pos_list_list)
