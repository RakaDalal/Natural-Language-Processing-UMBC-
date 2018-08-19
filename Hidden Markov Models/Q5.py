#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:51:42 2017

@author: rakadalal
"""

import json
import math
import operator
from scipy.misc import logsumexp

def logadd(lp, lq):
    return logsumexp([lp, lq])

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

def development(filename1,filename2):
    c=0
    word_list=[]
#    with open(filename1) as data_file:    
#        data = json.load(data_file)
#    for item in data:
#        for i in range(0,len(data[item])):
#            word=data[item][i]['text']
#            word_list.append(word)
        
    with open(filename2) as data_file:    
        data = json.load(data_file)
    for item in data:
        for i in range(0,len(data[item])):
            word=data[item][i]['text']
            word_list.append(word)
            c+=1
        if c>=10000:
            break
            
    return (word_list)

def test(filename):
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

def emission_transition(word_list, tokens, tags, emission_dict, transition_dict, alpha):
    emission_prob={}
    transition_prob={}
    for t in tags:
        previous="BOS"
        transition_prob[t]={}
        if t in transition_dict[previous]:
            upper=transition_dict[previous][t]+alpha
        else:
            upper=alpha
        lower=sum(transition_dict[previous].values())+(alpha*len(tags))
        tprob=upper/float(lower)
        transition_prob[t][previous]=math.log(tprob)
        for old in tags:
            if t in transition_dict[old]:
                upper=transition_dict[old][t]+alpha
            else:
                upper=alpha
            lower=sum(transition_dict[old].values())+(alpha*len(tags))
            tprob=upper/float(lower)
            transition_prob[t][old]=math.log(tprob)
    
    for i in range(0,len(word_list)):
            emission_prob[word_list[i]]={}
            for t in tags:
                if word_list[i] in emission_dict[t]:
                    upper=emission_dict[t][word_list[i]]+alpha
                else:
                    upper=alpha
                lower=sum(emission_dict[t].values())+(alpha*len(tokens))
                eprob=upper/float(lower)
                emission_prob[word_list[i]][t]=math.log(eprob)
                
    return (emission_prob, transition_prob)
                
            

def forward_algorithm(word_list, tokens, tags, emission_prob, transition_prob):
    a={}
    a[-1]={}
    tags=list(tags)
    for t in tags:
        a[-1][t]=0.0
    for i in range(0,len(word_list)):
        previous="BOS"
        a[i]={}
        for t in tags:
            eprob=emission_prob[word_list[i]][t]
            if i==0:
                tprob=transition_prob[t][previous]
                a[i][t]=eprob+tprob
            else:
                old=tags[0]
                tprob=transition_prob[t][old]
                a[i][t]=a[i-1][old]+eprob+tprob
                for j in range(1,len(tags)):
                    old=tags[j]
                    tprob=transition_prob[t][old]
                    a[i][t]=logadd(a[i][t], a[i-1][old]+eprob+tprob)
        if i%1000==0:
            print a[i][t]
            print str(i)+" "+str(t)
    return a    

def backward_algorithm(word_list, tokens, tags, emission_prob, transition_prob):
    tags=list(tags)
    c=len(word_list)-1
    b={}
    b[c]={}
    for t in tags:
        b[c][t]=0.0
    for i in range(len(word_list)-2,-2,-1):
            b[i]={}
            for j in range(0,len(tags)):
                t=tags[j]
                eprob=emission_prob[word_list[i+1]][t]
                for state in tags:
                    tprob=transition_prob[t][state]
                    if j==0:
                        b[i][state]=b[i+1][t]+eprob+tprob
                    else:
                        b[i][state]=logadd(b[i][state], b[i+1][t]+eprob+tprob)
            if i%1000==0:
                print b[i][t]
                print str(i)+" "+str(t)       
    return b  

def EM_algo(word_list, tokens, tags, emission_prob, transition_prob,a, b):
    c=len(word_list)-1
    tags=list(tags)
    to=len(tags)
    L=a[c][tags[to-1]]
    c_trans={}
    for t in tags:
        c_trans[t]={}
        for t2 in tags:
            c_trans[t][t2]=-float("inf")
    c_em={}
    #computing fractional counts
    for i in range(len(word_list)-2,-2,-1):
        c_em[word_list[i+1]]={}
        for t in tags:
            c_em[word_list[i+1]][t]=emission_prob[word_list[i+1]][t]+(a[i+1][t]+b[i+1][t]-L)
            for state in tags:
                u = emission_prob[word_list[i+1]][t] + transition_prob[t][state]
                c_trans[t][state]=logadd(c_trans[t][state],(a[i][state]+u+b[i+1][t]-L)) 
    totalTransition={}
    totalSubstitution={}
    #normalizing the counts to create new probabilities
    for t in tags:
        totalTransition[t]=-float("inf")
        totalSubstitution[t]=-float("inf") 
        for t2 in tags:
            totalTransition[t]=logadd(totalTransition[t],c_trans[t2][t])
        for i in range(len(word_list)-2,-2,-1):
            totalSubstitution[t]=logadd(totalSubstitution[t],c_em[word_list[i+1]][t])
    for t in tags:
        for t2 in tags:
            transition_prob[t2][t]=c_trans[t2][t]-totalTransition[t]
        for i in range(len(word_list)-2,-2,-1):
            emission_prob[word_list[i+1]][t]=c_em[word_list[i+1]][t]-totalSubstitution[t]
    
    return (emission_prob, transition_prob)
    
def viterbi(word_list_list, tokens, tags, emission_prob, transition_prob, alpha):
    predicted_list_list=[]
    for i in range(0,len(word_list_list)):
        v={}
        m={}
        predicted_list=[]
        previous="BOS"
        for j in range(0,len(word_list_list[i])):
            v[j]={}
            m[j]={}
            for t in tags:
                if word_list_list[i][j] in emission_prob:
                    try:
                        eprob=((1-alpha)*math.exp(emission_prob[word_list_list[i][j]][t]))+(alpha/len(tokens))
                    except:
                        print (emission_prob[word_list_list[i][j]][t])
                else:
                    eprob=alpha/len(tokens)
                if j==0:
                    if previous in transition_prob[t]:
                        tprob=((1-alpha)*math.exp(transition_prob[t][previous]))+(alpha/len(tags))
                    else:
                        tprob=alpha/len(tags)
                    v[j][t]=eprob*tprob
                    m[j][t]=previous
                else:
                    v[j][t]=-1
                    for old in tags:
                        try:
                            tprob=((1-alpha)*math.exp(transition_prob[t][old]))+(alpha/len(tags))
                        except:
                            print transition_prob[t][old] 
                        if ((v[j-1][old]*eprob*tprob)>v[j][t]):
                            v[j][t]=v[j-1][old]*eprob*tprob
                            m[j][t]=old
        k=len(word_list_list[i])-1   
        pos_tag=max(v[len(word_list_list[i])-1].iteritems(), key=operator.itemgetter(1))[0]
        predicted_list.append(pos_tag)
        while (k>0):
            pos_tag=m[k][pos_tag]
            predicted_list.append(pos_tag)
            k-=1
        predicted_list.reverse()
        predicted_list_list.append(predicted_list)
    return predicted_list_list  

def posterior(word_list_list, tokens, tags, emission_prob, transition_prob):
    predicted_list_list=[]
    for i in range(0,len(word_list_list)):
        predicted_list=[]
        for j in range(0,len(word_list_list[i])):
            maximum=0
            tag=""
            for t in tags:
                p=emission_prob[word_list_list[i][j]][t]
                p=math.exp(p)
                if p>maximum:
                    maximum=p
                    tag=t
            predicted_list.append(tag)
        predicted_list_list.append(predicted_list)
        
    return predicted_list_list
            
                 
def perplexity(word_list_list, pos_list_list, emission_prob, transition_prob, tokens, tags, alpha):
    J=0
    for i in range(0,len(word_list_list)):
        previous="BOS"
        prob=0.0
        for j in range(0,len(word_list_list[i])):
            if word_list_list[i][j] in emission_prob:
                eprob=((1-alpha)*math.exp(emission_prob[word_list_list[i][j]][pos_list_list[i][j]]))+(alpha/len(tokens))
            else:
                eprob=alpha/len(tokens)
            if previous in transition_prob[pos_list_list[i][j]]:
                tprob=((1-alpha)*math.exp(transition_prob[pos_list_list[i][j]][previous]))+(alpha/len(tags))
            else:
                tprob=alpha/len(tags)
            previous=pos_list_list[i][j]
            prob=prob+math.log(eprob)+math.log(tprob)
        J+=(prob)
    J=J/float(len(word_list_list))
    J=-J
    J=math.exp(J)
    return J
    
def evaluation(true_pos_list_list, predicted_pos_list_list):
    per_token_accuracy=0
    per_sentence_accuracy=0
    total_tokens=0
    for i in range(0,len(predicted_pos_list_list)):
        flag="True"
        for j in range(0,len(predicted_pos_list_list[i])):
            if predicted_pos_list_list[i][j]==true_pos_list_list[i][j]:
                per_token_accuracy+=1
            else:
                flag="False"
            total_tokens+=1
        if flag=="True":
            per_sentence_accuracy+=1
    per_token_accuracy=per_token_accuracy/float(total_tokens)
    per_sentence_accuracy=per_sentence_accuracy/float(len(predicted_pos_list_list))
    print ("Per token accuracy:"+str(per_token_accuracy))
    print ("Per sentence accuracy:"+str(per_sentence_accuracy))    
    

word_list_list, pos_list_list, tags,tokens=pos_vocabulary('data/train.json')
emission_dict, transition_dict=counts_from_training(word_list_list, pos_list_list, tags)
word_list=development('data/train.json','data/raw.json')
(emission_prob, transition_prob)=emission_transition(word_list, tokens, tags, emission_dict, transition_dict, 0.3)
word_list_list, true_pos_list_list=test('data/test.json')
posterior_list=[]
for i in range(0,len(word_list_list)):
    for j in range(0,len(word_list_list[i])):
        posterior_list.append(word_list_list[i][j])

for i in range(0,10):
    a=forward_algorithm(word_list, tokens, tags, emission_prob, transition_prob)
    b=backward_algorithm(word_list, tokens, tags, emission_prob, transition_prob)
    emission_prob, transition_prob=EM_algo(word_list, tokens, tags, emission_prob, transition_prob,a,b)
    predicted_pos_list_list=viterbi(word_list_list, tokens, tags, emission_prob, transition_prob, 0.3)
    print ("VITERBI:\n")
    evaluation(true_pos_list_list, predicted_pos_list_list)
    p=perplexity(word_list_list, true_pos_list_list, emission_prob, transition_prob, tokens, tags, 0.3)
    print ("Perplexity:"+str(p))
    
(emission_prob, transition_prob)=emission_transition(posterior_list, tokens, tags, emission_dict, transition_dict, 0.3) 
predicted_pos_list_list=posterior(word_list_list, tokens, tags, emission_prob, transition_prob)
print ("POSTERIOR:\n")
evaluation(true_pos_list_list, predicted_pos_list_list)
   

    