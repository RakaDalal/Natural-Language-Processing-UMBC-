#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:44:18 2017

@author: rakadalal
"""

import json
import autograd.numpy as np
from autograd import grad
from scipy import optimize
from sklearn import metrics
import random


#global variables
words_vocab={}
words_label_vocab={}
words_list=[]
words_label_list=[]
lang_count={}
character_vocab={}
character_bigram_vocab={}

character_bigram_features={}
label_bigram_features={}
model2_features={}
label_bigram_vocab={}
label_unigram_vocab={}
label_list=[]
lang_bigram_count={}
word_label={}
r_list=[]

#This method creates all the vocabularies from the training data
def character_vocabulary(filename):
    character_vocab={}
    character_bigram_vocab={}
    label_unigram_vocab={}
    label_bigram_vocab={}
    with open(filename) as data_file:    
        data = json.load(data_file)
    for item in data:
        for i in range(0,len(data[item])):
            lang=data[item][i]['lang']
            text=data[item][i]['text']
            for j in range(0,len(text)):
                if text[j] in character_vocab:
                    character_vocab[text[j]]+=1
                else:
                    character_vocab[text[j]]=1
            
            for j in range(0,len(text)-1):
                bigram=text[j]+text[j+1]
                if bigram in character_bigram_vocab:
                    character_bigram_vocab[bigram]+=1
                else:
                    character_bigram_vocab[bigram]=1
                if lang in lang_bigram_count:
                    lang_bigram_count[lang]+=1
                else:
                    lang_bigram_count[lang]=1
                tup=(bigram,lang)
                tup2=(text[j],lang)
                if tup2 in label_unigram_vocab:
                    label_unigram_vocab[tup2]+=1
                else:
                    label_unigram_vocab[tup2]=1
                if tup in label_bigram_vocab:
                    label_bigram_vocab[tup]+=1
                else:
                    label_bigram_vocab[tup]=1
    
    return (character_vocab,character_bigram_vocab,label_bigram_vocab,label_unigram_vocab,lang_bigram_count)

#This method creates dictionaries to help compute features for model3
def saving_dictionaries():
    character_bigram_features={}
    label_bigram_features={}
    V=len(character_vocab)                                  
    for key in character_bigram_vocab:
        upper=np.float64(character_bigram_vocab[key])
        lower=np.float64(character_vocab[key[0]])
        prob=(upper+1)/(np.float64(lower+V))
        character_bigram_features[key]=prob
                                 
    for label in label_list:
        for key in character_bigram_vocab:
            tup=(key,label)
            if tup in label_bigram_vocab:
                value=label_bigram_vocab[tup]/(np.float64(character_bigram_vocab[key]))
                label_bigram_features[tup]=value
            else:
                label_bigram_features[tup]=(np.float64(0))
    return (character_bigram_features,label_bigram_features)

#This method creates dictionaries to help compute features for model2                
def dictionaries_model2():
    bigrams_list=[]
    model2_features={}
    for key in character_bigram_vocab:
        bigrams_list.append(key)
    for label in label_list:
        for key in character_bigram_vocab:
            tup=(key,label)
            tup2=(key[0],label)
            if tup in label_bigram_vocab:
                value=label_bigram_vocab[tup]/(np.float64(label_unigram_vocab[tup2]))
                model2_features[tup]=value
            else:
                model2_features[tup]=(np.float64(0))
    return (model2_features,bigrams_list)

#This method creates all the vocabularies required for model 1 from the training data
def train_json_to_dict(filename):
    words_vocab={}
    words_label_vocab={}
    labels=[]
    lang_count={}
    with open(filename) as data_file:    
        data = json.load(data_file)   
    for item in data:
        for i in range(0,len(data[item])):
            lang=data[item][i]['lang']
            labels.append(lang)
            text=data[item][i]['text']
            tup=(lang,text)
            if tup in words_label_vocab:
                words_label_vocab[tup]+=1
            else:
                words_label_vocab[tup]=1
            if text in words_vocab:
                words_vocab[text]+=1
            else:
                words_vocab[text]=1
            if lang in lang_count:
                lang_count[lang]+=1
            else:
                lang_count[lang]=1
    labels=set(labels)
    labels=list(labels)
    return (words_vocab,words_label_vocab,labels,lang_count)

#This method reads the word-lang pairs and words from dev data
def json_to_list(filename):
    words_list=[]
    words_label_list=[]
    with open(filename) as data_file:    
        data = json.load(data_file)   
    for item in data:
        for i in range(0,len(data[item])):
            lang=data[item][i]['lang']
            text=data[item][i]['text']
            tup=(lang,text)
            words_label_list.append(tup)
            words_list.append(text)
    
    return (words_list,words_label_list)

#This method computes p(bigram|lang) for model 2
def prob_model2(bigram,lang,param):
    if bigram not in bigrams_list:
        value=np.exp(0)
    else:
        tup=(bigram,lang)
        pos_l=label_list.index(lang)*len(character_bigram_vocab)
        pos_k=bigrams_list.index(bigram)
        pos=pos_l+pos_k
        if tup in model2_features:
            value=param[pos]*model2_features[tup]
        else:
            value=np.float64(0)
        value=np.exp(value)
    lower=0
    for key in character_vocab:
        try:
            new_bigram=bigram[0]+key
            tup=(new_bigram,lang)
            pos_k=bigrams_list.index(new_bigram)
            pos=pos_l+pos_k
            if tup in model2_features:
                new_value=param[pos]*model2_features[tup]
            else:
                new_value=np.float64(0)
            new_value=np.exp(new_value)
            lower+=new_value
        except:
            continue
    if lower==0:
        prob=1
    else:
        prob=value/lower
                         
    return prob

#This method computes the loglikelihood for model2
def model2(param):
    N=len(r_list)
    J=0
    total=np.float64(sum(lang_count.values()))
    for i in r_list:
        lang=words_label_list[i][0]
        word=words_label_list[i][1]
        prob=lang_count[lang]/total
        for j in range(0,len(word)-1):
            bigram=word[j]+word[j+1]
            prob_bi=prob_model2(bigram,lang,param)
            prob=prob*prob_bi
        if prob>1:
            print prob
        J+=np.log(prob)
    J=J/N
    return -J 

#This method computes the gradient for model2
def model2_grad(param):
   return grad(model2)(param) 

#This method predicts labels of test data using model2
def prediction_model2(filename,label_list,param):
    total=np.float64(sum(lang_count.values()))
    true_labels=[]
    words=[]
    predicted_labels=[]
    with open(filename) as data_file:    
        data = json.load(data_file) 
    counter=0
    for item in data:
        for i in range(0,len(data[item])):
            lang=data[item][i]['lang']
            true_labels.append(lang)
            text=data[item][i]['text']
            words.append(text)
            counter+=1
            
        if counter>200:
            break
    
    for item in words:
        maximum=0
        predict=""
        for label in label_list:
            prob=lang_count[label]/total
            for j in range(0,len(item)-1):
                bigram=item[j]+item[j+1]
                prob_bi=prob_model2(bigram,label,param)
                prob=prob*prob_bi
            if prob>1:
                print prob
            if prob > maximum:
                maximum=prob
                predict=label
        print predict
        predicted_labels.append(predict)
    return (true_labels,predicted_labels)
                    
            
            
    
#This method creates a list of random numbers to select data from dev file randomly and optimize the parameters using that   
def random_list(N):
    for x in range(N):
        i=random.randint(0,8000)
        r_list.append(i)
    return r_list

#This method computes exp(theta_transpose.g(z,l)) for a given word,language pair  
def prob_model3(word,lang,param):
    
    start=label_list.index(lang)*len(character_bigram_vocab)*2
    i=start
    value=0
    for key in character_bigram_vocab:
        if key in word:
            tup=(key,lang)
            value+=(param[i]*label_bigram_features[tup])
            i+=1
            value+=(param[i]*character_bigram_features[key])
            i+=1
    value=np.exp(value)
    return value

#This method computes the loglikelihood for model3
def model3(param):
    param=np.array(param)
    N=200
    J=0
    for i in r_list:
        lang=words_label_list[i][0]
        word=words_label_list[i][1]
        prob_w=prob_model3(word,lang,param)
        lower=0
        for label in label_list:
            
            prob_w_l=prob_model3(word,label,param)
            lower+=prob_w_l
        prob=(prob_w)/(lower)
#        print prob
        if prob > 1:
            print prob
        J+=np.log(prob)
    J=J/N
    J=J-(0.01*np.linalg.norm(param))
    return -J

#This method computes the gradient for model3                
def model3_grad(param):
   return grad(model3)(param) 

#This method predicts labels of test data using model3
def prediction_model3(filename,label_list,param):
    true_labels=[]
    words=[]
    predicted_labels=[]
    with open(filename) as data_file:    
        data = json.load(data_file)   
    for item in data:
        for i in range(0,len(data[item])):
            lang=data[item][i]['lang']
            true_labels.append(lang)
            text=data[item][i]['text']
            words.append(text)
    for item in words:
        maximum=0
        predict=""
        for label in label_list:
            prob=prob_model3(item,label,param)
            if prob > maximum:
                maximum=prob
                predict=label
        predicted_labels.append(predict)
    return (true_labels,predicted_labels)  

#This method predicts labels of test data using model1
def prediction_model1(filename,label_list,param):
    V=len(words_vocab)
    true_labels=[]
    words=[]
    predicted_labels=[]
    with open(filename) as data_file:    
        data = json.load(data_file)   
    for item in data:
        for i in range(0,len(data[item])):
            lang=data[item][i]['lang']
            true_labels.append(lang)
            text=data[item][i]['text']
            words.append(text)
    for item in words:
        maximum=0
        predict=""
        for label in label_list:
            tup=(label,item)
            if tup in words_label_vocab:
                upper=words_label_vocab[tup]
            else:
                upper=0
            if item in words_vocab:
                lower=words_vocab[item]
            else:
                lower=0
            prob=(upper+param)/(float(lower+(param*V)))
            if prob > maximum:
                maximum=prob
                predict=label
        predicted_labels.append(predict)
    return (true_labels,predicted_labels)
            

#This method computes the loglikelihood for model1
def model1(param):
    N=len(words_label_list)
    V=len(words_vocab)
    J=0
    for i in range(0,len(words_label_list)):
        if words_label_list[i] in words_label_vocab:
            upper=np.float64(words_label_vocab[words_label_list[i]])
        else:
            upper=np.float64(0)
        if words_list[i] in words_vocab:
            lower=np.float64(words_vocab[words_list[i]])
        else:
            lower=np.float64(0)
        prob=(upper+np.exp(param))/((lower+(np.exp(param)*V)))
        J+=np.log(prob)
    J=J/N
    return -J               

#This method computes the gradient for model1
def model1_grad(param):
    return grad(model1)(param)
                

#This method computes the micro-recall, micro_precision, macro_recall and macro_precision
def evaluation(true_labels,predicted_labels):
    micro_recall=metrics.recall_score(true_labels,predicted_labels,average = "micro")
    micro_precision=metrics.precision_score(true_labels,predicted_labels,average = "micro")
    macro_recall=metrics.recall_score(true_labels,predicted_labels,average = "macro")
    macro_precision=metrics.precision_score(true_labels,predicted_labels,average = "macro")
    
    return (micro_recall, micro_precision, macro_recall, macro_precision)
    
    
    
words_vocab,words_label_vocab,label_list,lang_count=train_json_to_dict('train.json')
words_list,words_label_list=json_to_list('dev.json')

#model1
res = optimize.minimize(model1,np.ones(1), method='L-BFGS-B',jac=model1_grad,options={'gtol': 1e-4, 'disp': True})
param=np.exp(res.x)[0]
true_labels,predicted_labels=prediction_model1('test.json',label_list,param)
print evaluation(true_labels,predicted_labels)

character_vocab,character_bigram_vocab,label_bigram_vocab,label_unigram_vocab,lang_bigram_count=character_vocabulary('train.json')
character_bigram_features,label_bigram_features=saving_dictionaries()
model2_features,bigrams_list=dictionaries_model2()
r_list=random_list(200)

#model2
res = optimize.minimize(model2,np.ones(32301), method='L-BFGS-B',jac=model2_grad,options={'gtol': 1e-3, 'disp': True})
param=(res.x)
true_labels,predicted_labels=prediction_model2('test.json',label_list,param)
print evaluation(true_labels,predicted_labels)

#model3
res = optimize.minimize(model3,np.ones(64602), method='L-BFGS-B',jac=model3_grad,options={'gtol': 1e-3, 'disp': True})
param=(res.x)
true_labels,predicted_labels=prediction_model3('test.json',label_list,param)
print evaluation(true_labels,predicted_labels)
