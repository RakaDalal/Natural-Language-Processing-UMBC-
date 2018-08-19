#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:32:00 2017

@author: rakadalal
"""

from nltk.util import ngrams
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.table import Table
import gzip

#This method creates the vocabulary    
def vocabCreate(filename):
    with gzip.open(filename, 'rb') as filepointer:
        filep=filepointer.readlines()
    vocabulary={}
    for line in filep:
        line=line.strip("\n")
        line=line.split(" ")
        for item in line:
            if item.strip() in vocabulary:
                vocabulary[(item.strip())]+=1
            else:
                vocabulary[(item.strip())]=1 
    return vocabulary

#This method creates unigram from the file provided and stores them in a dictionary along with the counts
def unigramscount(filename,oov):
    unigram_dict={}
    with gzip.open(filename, 'rb') as filepointer:
        filep=filepointer.readlines()
    for line in filep:
        line=line.strip("\n")
        line2=line
        line=line.split(" ")
        for item in line:
            if item in oov:
                try:
                    line2=str.replace(str(line2),str(item),"oov")
                except:
                    continue
        line2="<START> <START> "+line2
        line2=line2+" <END> <END>"
        line2=line2.split(" ")
        unigrams = ngrams(line2,1)
        for item in unigrams:
            if item in unigram_dict:
                unigram_dict[item]+=1
            else:
                unigram_dict[item]=1
    return unigram_dict

#This method creates bigram from the file provided and stores them in a dictionary along with the counts
def bigramscount(filename,oov):
    bigram_dict={}
    with gzip.open(filename, 'rb') as filepointer:
        filep=filepointer.readlines()
    for line in filep:
        line=line.strip("\n")
        line2=line
        line=line.split(" ")
        for item in line:
            if item in oov:
                try:
                    line2=str.replace(str(line2),str(item),"oov")
                except:
                    continue
        line2="<START> <START> "+line2
        line2=line2+" <END> <END>"
        line2=line2.split(" ")
        bigrams = ngrams(line2,2)
        for item in bigrams:
            if item in bigram_dict:
                bigram_dict[item]+=1
            else:
                bigram_dict[item]=1
    return bigram_dict

#This method creates trigram from the file provided and stores them in a dictionary along with the counts
def trigramscount(filename,oov):
    trigram_dict={}
    with gzip.open(filename, 'rb') as filepointer:
        filep=filepointer.readlines()
    for line in filep:
        line=line.strip("\n")
        line2=line
        line=line.split(" ")
        for item in line:
            if item in oov:
                try:
                    line2=str.replace(str(line2),str(item),"oov")
                except:
                    continue
        line2="<START> <START> "+line2
        line2=line2+" <END> <END>"
        line2=line2.split(" ")
        trigrams = ngrams(line2,3)
        for item in trigrams:
            if item in trigram_dict:
                trigram_dict[item]+=1
            else:
                trigram_dict[item]=1
    return trigram_dict

#This methods finds 300 least frequent words and mark them as out of vocabulary(oov) 
def least_common_words(vocabulary):
    oov=[]
    count=0
    for key, value in sorted(vocabulary.iteritems(), key=lambda (k,v): (v,k)):
        count+=1
        if count<300:
            oov.append(key)
    return oov

#This method performs counting from the development or test file provided and stores them in dictionaries along with the counts and also creates sets (one each for unigram, bigram and trigram)
def counting(filename):
    unigram_list_total=[]
    bigram_list_total=[]
    trigram_list_total=[]
    unigram_set=[]
    bigram_set=[]
    trigram_set=[]
    with gzip.open(filename, 'rb') as filepointer:
        filep=filepointer.readlines()
    for line in filep:
        line=line.strip("\n")
        line="<START> <START> "+line
        line=line+" <END> <END>"
        line=line.split(" ")
        unigrams = ngrams(line,1)
        unigram_list=[]
        for item in unigrams:
                unigram_list.append(item)
                unigram_set.append(item)
        unigram_list_total.append(unigram_list)
        bigrams = ngrams(line,2)
        bigram_list=[]
        for item in bigrams:
                bigram_list.append(item)
                bigram_set.append(item)
        bigram_list_total.append(bigram_list)
        trigrams = ngrams(line,3)
        trigram_list=[]
        for item in trigrams:
                trigram_list.append(item)
                trigram_set.append(item)
        trigram_list_total.append(trigram_list)
    unigram_set=set(unigram_set)
    bigram_set=set(bigram_set)
    trigram_set=set(trigram_set)
    tup=(unigram_list_total,bigram_list_total,trigram_list_total,unigram_set,bigram_set,trigram_set)
    return (tup)

#This method computes the perplexity for laplace trigram model
def laplace_trigram(bigram_list,trigram_list,bigram_dict,trigram_dict,param):
    value=0
    count=0
    for i in range(0, len(bigram_list)):
        for j in range(0, len(trigram_list[i])):
            if trigram_list[i][j] in trigram_dict:
                upper=trigram_dict[trigram_list[i][j]]+param
            else:
                upper=2+param
            if bigram_list[i][j] in bigram_dict:
                lower=bigram_dict[bigram_list[i][j]]+(len(bigram_dict)*param)
            else:
                lower=bigram_dict[(u'oov', u'oov')]+(len(bigram_dict)*param)
            prob=upper/float(lower)
            value+=math.log(prob)
            count+=1
    value=value/float(count)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)

#This method computes the perplexity for laplace bigram model   
def laplace_bigram(unigram_list,bigram_list,unigram_dict,bigram_dict,param):
    value=0
    count=0
    for i in range(0, len(unigram_list)):
        for j in range(0, len(bigram_list[i])):
            if bigram_list[i][j] in bigram_dict:
                upper=bigram_dict[bigram_list[i][j]]+param
            else:
                upper=bigram_dict[(u'oov', u'oov')]+param
            if unigram_list[i][j] in unigram_dict:
                lower=unigram_dict[unigram_list[i][j]]+(len(unigram_dict)*param)
            else:
                lower=unigram_dict[(u'oov',)]+(len(unigram_dict)*param)
            prob=upper/float(lower)
            value+=math.log(prob)
            count+=1
    value=value/float(count)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)

#This method computes the perplexity for laplace unigram model    
def laplace_unigram(unigram_list,unigram_dict,param):
    value=0
    count=0
    total=sum(unigram_dict.values())
    for i in range(0, len(unigram_list)):
        for j in range(0, len(unigram_list[i])):
            if unigram_list[i][j] in unigram_dict:
                upper=unigram_dict[unigram_list[i][j]]+param
            else:
                upper=unigram_dict[(u'oov',)]+param
            lower=total+(len(unigram_dict)*param)
            prob=upper/float(lower)
            value+=math.log(prob)
            count+=1
    value=value/float(count)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)

#This method computes the perplexity for interpolation trigram model    
def interpolate_trigram(unigram_list, bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,param1,param2,param3):
    value=0
    count=0
    total=sum(unigram_dict.values())
    for i in range(0, len(bigram_list)):
        for j in range(0, len(trigram_list[i])):
            if trigram_list[i][j] in trigram_dict:
                upper1=trigram_dict[trigram_list[i][j]]
            else:
                upper1=0
            if bigram_list[i][j] in bigram_dict:
                lower1=bigram_dict[bigram_list[i][j]]
            else:
                lower1=bigram_dict[(u'oov', u'oov')]
            if bigram_list[i][j+1] in bigram_dict:
                upper2=bigram_dict[bigram_list[i][j+1]]
            else:
                upper2=bigram_dict[(u'oov', u'oov')]
            if unigram_list[i][j+1] in unigram_dict:
                lower2=unigram_dict[unigram_list[i][j+1]]
            else:
                lower2=unigram_dict[(u'oov',)]
            if unigram_list[i][j+2] in unigram_dict:
                upper3=unigram_dict[unigram_list[i][j+2]]
            else:
                upper3=unigram_dict[(u'oov',)]
            lower3=total
            prob=(param1*(upper1/float(lower1)))+(param2*(upper2/float(lower2)))+(param3*(upper3/float(lower3)))+((1-param1-param2-param3)*(1/float(len(unigram_dict))))
            value+=math.log(prob)
            count+=1
    value=value/float(count)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)

#This method computes the perplexity for interpolation bigram model 
def interpolate_bigram(unigram_list, bigram_list,unigram_dict,bigram_dict,param1,param2):
    value=0
    count=0
    total=sum(unigram_dict.values())
    for i in range(0, len(bigram_list)):
        for j in range(0, len(bigram_list[i])):
            if bigram_list[i][j] in bigram_dict:
                upper1=bigram_dict[bigram_list[i][j]]
            else:
                upper1=bigram_dict[(u'oov', u'oov')]
            if unigram_list[i][j] in unigram_dict:
                lower1=unigram_dict[unigram_list[i][j]]
            else:
                lower1=unigram_dict[(u'oov',)]
            if unigram_list[i][j+1] in unigram_dict:
                upper2=unigram_dict[unigram_list[i][j+1]]
            else:
                upper2=unigram_dict[(u'oov',)]
            lower2=total
            prob=(param1*(upper1/float(lower1)))+(param2*(upper2/float(lower2)))+((1-param1-param2)*(1/float(len(unigram_dict))))
            value+=math.log(prob)
            count+=1
    value=value/float(count)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)

#This method computes the perplexity for interpolation unigram model     
def interpolate_unigram(unigram_list,unigram_dict,param):
    value=0
    count=0
    total=sum(unigram_dict.values())
    for i in range(0, len(unigram_list)):
        for j in range(0, len(unigram_list[i])):
            if unigram_list[i][j] in unigram_dict:
                upper=unigram_dict[unigram_list[i][j]]
            else:
                upper=unigram_dict[(u'oov',)]
            lower=total
            prob=(param*(upper/float(lower)))+((1-param)*(1/float(len(unigram_dict))))
            value+=math.log(prob)
            count+=1
    value=value/float(count)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)

##This method is used for 2D plotting 
def plot(X,Y,x_label,y_label):
    plt.plot(X, Y)
    plt.show()
    
def plot_3(X,Y,Z,x_label,y_label,z_label):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(X,Y,Z,zdir='z')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    plt.show()
 
#This method evaluates the laplace model and finds the best parameter. It also plots the evaluation 
def evaluation_laplace(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,N):
    X=[]
    Y=[]
    i=0.5
    while (i<=5):
        if N==1:
            perplexity=laplace_unigram(unigram_list,unigram_dict,i)
        elif N==2:
            perplexity=laplace_bigram(unigram_list,bigram_list,unigram_dict,bigram_dict,i)
        else:
            perplexity=laplace_trigram(bigram_list,trigram_list,bigram_dict,trigram_dict,i)
        X.append(i)
        Y.append(perplexity)
        i+=0.5
    plot(X,Y,"parameter","perplexity")
    lowestPerplexity = (min(Y))
    ind=Y.index(lowestPerplexity)
    print ("The perplexity is: "+str(lowestPerplexity))
    return (X[ind])

 
#This method evaluates the interpolation unigram model and finds the best parameter. It also plots the evaluation    
def evaluation_interpolate_unigram(unigram_list,unigram_dict):
    X=[]
    Y=[]
    i=0
    while (i<=1):
        perplexity=interpolate_unigram(unigram_list, unigram_dict,i)
        X.append(i)
        Y.append(perplexity)
        i+=0.05
    plot(X,Y,"parameter","perplexity")
    lowestPerplexity = (min(Y))
    ind=Y.index(lowestPerplexity)
    print ("The perplexity is: "+str(lowestPerplexity))
    return (X[ind])

#This method evaluates the interpolation bigram model and finds the best parameters. It also plots the evaluation      
def evaluation_interpolate_bigram(unigram_list,bigram_list,unigram_dict,bigram_dict):
    X=[]
    Y=[]
    Z=[]
    i=0
    j=0
    while (i<=1):
        while(j<=1):
            if ((i+j)<=1):
                perplexity=interpolate_bigram(unigram_list, bigram_list, unigram_dict,bigram_dict,i,j)
                tup=(i,j)
                X.append(i)
                Y.append(j)
                Z.append(perplexity)
            j+=0.1
        i+=0.1
        j=0
    plot(X,Y,Z,"parameter1","parameter2","perplexity")
    lowestPerplexity = (min(Z))
    ind=Z.index(lowestPerplexity)
    tup=(X[ind],Y[ind])
    print ("The perplexity is: "+str(lowestPerplexity))
    return (tup)

#This method evaluates the interpolation trigram model and finds the best parameters. It also plots the evaluation     
def evaluation_interpolate_trigram(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict):
    X=[]
    Y=[]
    Z=[]
    P=[]
    i=0
    j=0
    k=0
    while (i<=1):
        while(j<=1):
            while (k<=0):
                if ((i+j+k)<=1):
                    perplexity=interpolate_trigram(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,i,j,k)
                    X.append(i)
                    Y.append(j)
                    Z.append(k)
                    P.append(perplexity)
                k+=0.1
            j+=0.1
            k=0
        i+=0.1
        j=0
        k=0
    t = Table([X, Y, Z, P], names=('parameter1', 'parameter2', 'parameter3', 'perplexity'))
    print (t)
    lowestPerplexity = (min(P))
    ind=P.index(lowestPerplexity)
    tup=(X[ind],Y[ind],Z[ind])
    print ("The perplexity is: "+str(lowestPerplexity))
    return (tup)

#This method evaluates the backoff unigram model and finds the best parameters. It also plots the evaluation 
def evaluation_backoff_unigram(unigram_list,unigram_dict,unigram_set):
    X=[]
    Y=[]
    Z=[]
    i=0
    j=0.1
    while (i<5):
        while (j<0.5):
            perplexity=backoff_unigram(unigram_list,unigram_dict,unigram_set,i,j)
            X.append(i)
            Y.append(j)
            Z.append(perplexity)
            j+=0.1
        i+=1
        j=0.1
    lowestPerplexity = (min(Z))
    ind=Z.index(lowestPerplexity)
    plot(X,Y,Z,"epsilon","discount1","perplexity")
    tup=(X[ind],Y[ind])
    print ("The perplexity is: "+str(lowestPerplexity))
    return (tup)

#This method evaluates the backoff bigram model and finds the best parameters. It also plots the evaluation 
def evaluation_backoff_bigram(unigram_list,bigram_list,unigram_dict,bigram_dict,unigram_set,bigram_set):
    X=[]
    Y=[]
    Z=[]
    P=[]
    i=0
    j=0.1
    k=0.1
    while (i<5):
        while (j<0.5):
            while (k<0.5):
                perplexity=backoff_bigram(unigram_list,bigram_list,unigram_dict,bigram_dict,unigram_set,bigram_set,i,j,k)
                X.append(i)
                Y.append(j)
                Z.append(k)
                P.append(perplexity)
                k+=0.2
            j+=0.2
            k=0.1
        i+=1
        j=0.1
        k=0.1
    t = Table([X, Y, Z, P], names=('epsilon', 'discount1', 'discount2', 'perplexity'))
    print (t)
    lowestPerplexity = (min(P))
    ind=P.index(lowestPerplexity)
    tup=(X[ind],Y[ind],Z[ind])
    print ("The perplexity is: "+str(lowestPerplexity))
    return (tup)
#This method evaluates the backoff trigram model and finds the best parameters. It also plots the evaluation 
def evaluation_backoff_trigram(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,unigram_set,bigram_set,trigram_set):
    X=[]
    Y=[]
    Z=[]
    R=[]
    P=[]
    i=0
    j=0.1
    k=0.1
    l=0.1
    while (i<=3):
        while (j<0.5):
            while (k<0.5):
                while (l<0.5):
                    perplexity=backoff_trigram(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,unigram_set,bigram_set,trigram_set,i,j,k,l)
                    X.append(i)
                    Y.append(j)
                    Z.append(k)
                    R.append(l)
                    P.append(perplexity)
                    l+=0.2
                k+=0.2
                l=0.1
            j+=0.2
            k=0.1
            l=0.1
        i+=1
        j=0.1
        k=0.1
        l=0.1
    t = Table([X, Y, Z, R, P], names=('epsilon', 'discount1', 'discount2', 'discount3','perplexity'))
    print (t)
    lowestPerplexity = (min(P))
    ind=P.index(lowestPerplexity)
    tup=(X[ind],Y[ind],Z[ind],R[ind])
    print ("The perplexity is: "+str(lowestPerplexity))
    return (tup)

#This method computes the beta for backoff     
def backoff_beta(unigram_list,unigram_dict,unigram_set,epsilon,discount1):
    total=sum(unigram_dict.values())
    c_y_d1=0
    c_y=0
    for item in unigram_set:
        if item in unigram_dict and unigram_dict[item]>epsilon:
            c_y_d1+=(unigram_dict[item]-discount1)
        else:
            c_y+=1
    beta=c_y_d1/float(total)
    beta=1-beta
    beta=beta/float(c_y)
    return (beta)

#This method computes the alpha for backoff 
def backoff_alpha(unigram_list,bigram_list,unigram_dict,bigram_dict,bigram_set,epsilon,discount1,discount2):
    
    alpha_utility={}
    for item in bigram_set:
        (x,y)=item
        z2=(y,)
        z1=(x,)
        c_x_y=0
        c_y_d1=0
        c_y=0
        if item in bigram_dict and bigram_dict[item]>epsilon:
            c_x_y=(bigram_dict[item]-discount2)
        else:
            if z2 in unigram_dict and unigram_dict[z2]>epsilon:
                c_y_d1=(unigram_dict[z2]-discount1)
            else:
                c_y=1
        tup=(c_x_y,c_y_d1,c_y)            
        if z1 in alpha_utility:
            actual_tup=alpha_utility[z1]
            tup2=(actual_tup[0]+tup[0],actual_tup[1]+tup[1],actual_tup[2]+tup[2])
            alpha_utility[z1]=tup2
        else:
            alpha_utility[z1]=tup
    return (alpha_utility)

#This method computes the gamma for backoff 
def backoff_gamma(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,trigram_set,epsilon,discount1,discount2,discount3):
    
    gamma_utility={}
    for item in trigram_set:
        (x,y,z)=item
        t3=(z,)
        t2=(y,z)
        t1=(x,y)
        c_x_y_z=0
        c_y_z=0
        c_z_d1=0
        c_z=0
        if item in trigram_dict and trigram_dict[item]>epsilon:
            c_x_y_z=(trigram_dict[item]-discount3)
        else:
            if t2 in bigram_dict and bigram_dict[t2]>epsilon:
                c_y_z=(bigram_dict[t2]-discount2)
            else:
                if t3 in unigram_dict and unigram_dict[t3]>epsilon:
                    c_z_d1=(unigram_dict[t3]-discount1)
                else:
                    c_z=1
        tup=(c_x_y_z,c_y_z,c_z_d1,c_z)            
        if t1 in gamma_utility:
            actual_tup=gamma_utility[t1]
            tup2=(actual_tup[0]+tup[0],actual_tup[1]+tup[1],actual_tup[2]+tup[2],actual_tup[3]+tup[3])
            gamma_utility[t1]=tup2
        else:
            gamma_utility[t1]=tup
    return (gamma_utility)

#This method computes the perplexity for backoff unigram model    
def backoff_unigram(unigram_list,unigram_dict,unigram_set,epsilon,discount1): 
    total=sum(unigram_dict.values())
    beta=backoff_beta(unigram_list,unigram_dict,unigram_set,epsilon,discount1)              
    value=0
    count=0
    for i in range(0, len(unigram_list)):
        for j in range(0, len(unigram_list[i])):
            if unigram_list[i][j] in unigram_dict and unigram_dict[unigram_list[i][j]]>epsilon:
                upper=unigram_dict[unigram_list[i][j]]-discount1
                lower=total
                prob=upper/float(lower)
            else:
                prob=beta
            value+=math.log(prob)
            count+=1
    value=value/float(count)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)

#This method computes the perplexity for backoff bigram model    
def backoff_bigram(unigram_list,bigram_list,unigram_dict,bigram_dict,unigram_set,bigram_set,epsilon,discount1,discount2): 
    total=sum(unigram_dict.values())
    beta=backoff_beta(unigram_list,unigram_dict,unigram_set,epsilon,discount1) 
    alpha_utility= backoff_alpha(unigram_list,bigram_list,unigram_dict,bigram_dict,bigram_set,epsilon,discount1,discount2)       
    value=0
    count=0
    for i in range(0, len(bigram_list)):
        for j in range(0, len(bigram_list[i])):
            if bigram_list[i][j] in bigram_dict and bigram_dict[bigram_list[i][j]]>epsilon:
                upper=bigram_dict[bigram_list[i][j]]-discount2
                lower=unigram_dict[unigram_list[i][j]]
                prob=upper/float(lower)
            else:
                if unigram_list[i][j] in unigram_dict:
                    c_x=unigram_dict[unigram_list[i][j]]
                else:
                    c_x=unigram_dict[(u'oov',)]
                (c_x_y,c_y_d1,c_y)=alpha_utility[unigram_list[i][j]]
                alpha_lower= (c_y_d1/float(total)) + (c_y*(beta/float(len(unigram_dict))))
                alpha=(1-(c_x_y/float(c_x)))/(float(alpha_lower))
                if unigram_list[i][j+1] in unigram_dict and unigram_dict[unigram_list[i][j+1]]>epsilon:
                    upper=unigram_dict[unigram_list[i][j+1]]-discount1
                    lower=total
                    prob=alpha*(upper/float(lower))
                else:
                    prob=alpha*(beta/float(len(unigram_dict)))
            value+=math.log(prob)
            count+=1
    value=value/float(count)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)         

#This method computes the perplexity for backoff trigram model
def backoff_trigram(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,unigram_set,bigram_set,trigram_set,epsilon,discount1,discount2,discount3): 
    total=sum(unigram_dict.values())
    beta=backoff_beta(unigram_list,unigram_dict,unigram_set,epsilon,discount1) 
    alpha_utility= backoff_alpha(unigram_list,bigram_list,unigram_dict,bigram_dict,bigram_set,epsilon,discount1,discount2)       
    gamma_utility= backoff_gamma(unigram_list,bigram_list,trigram_list,unigram_dict,bigram_dict,trigram_dict,trigram_set,epsilon,discount1,discount2,discount3)
    value=0
    count=0
    for i in range(0, len(trigram_list)):
        for j in range(0, len(trigram_list[i])):
            if trigram_list[i][j] in trigram_dict and trigram_dict[trigram_list[i][j]]>epsilon:
                upper=trigram_dict[trigram_list[i][j]]-discount3
                lower=bigram_dict[bigram_list[i][j]]
                prob=upper/float(lower)
            else:
                if bigram_list[i][j] in bigram_dict:
                    c_x_y=bigram_dict[bigram_list[i][j]]
                else:
                    c_x_y=bigram_dict[(u'oov',u'oov')] 
                if unigram_list[i][j+1] in unigram_dict:
                    c_y=unigram_dict[unigram_list[i][j+1]]
                else:
                    c_y=unigram_dict[(u'oov',)]
                (c_x_y_dash,c_y_d1,c_y_dash)=alpha_utility[unigram_list[i][j+1]]
                alpha_lower= (c_y_d1/float(total)) + (c_y_dash*(beta/float(len(unigram_dict))))
                if (alpha_lower==0):
                    alpha_lower=0.01
                alpha=(1-(c_x_y_dash/float(c_y)))/(float(alpha_lower))
                (c_x_y_z,c_y_z,c_z_d1,c_z)=gamma_utility[bigram_list[i][j]]
                gamma_lower=(c_y_z/float(c_y))+(alpha*(c_z_d1/float(total)))+(alpha*(c_z*(beta/float(len(unigram_dict)))))
                gamma=(1-(c_x_y_z/float(c_x_y)))/float(gamma_lower)
                if bigram_list[i][j+1] in bigram_dict and bigram_dict[bigram_list[i][j+1]]>epsilon:
                    upper=bigram_dict[bigram_list[i][j+1]]-discount2
                    lower=unigram_dict[unigram_list[i][j+1]]
                    prob=gamma*(upper/float(lower))
                else:
                    if unigram_list[i][j+2] in unigram_dict and unigram_dict[unigram_list[i][j+2]]>epsilon:
                        upper=unigram_dict[unigram_list[i][j+2]]-discount1
                        lower=total
                        prob=gamma*alpha*(upper/float(lower))
                    else:
                        prob=gamma*alpha*(beta/float(len(unigram_dict)))
                
            value+=math.log(prob)
            count+=1
    value=value/float(count)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)

#This method computes the perplexity for zerogram model
def zerogram(unigram_dict):
    prob=1/float(len(unigram_dict))
    value=math.log(prob)
    value=value * (-1)
    perplexity=math.exp(value)
    return (perplexity)  


                                