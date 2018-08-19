#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:51:42 2017

@author: rakadalal
"""

import numpy as np
import scipy
import math
import codecs
from collections import defaultdict
from matplotlib import pyplot as plt
import operator


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def counts(path):
    words_tag=defaultdict(list)
    words_tag_short=defaultdict(list)
    words_tag_position=defaultdict(list)
    tag_count={};tag_short_count={}
    filepointer=codecs.open(path,'r',encoding='utf-8')
    filep=filepointer.readlines()
    i=2
    flag=0
    while i< len(filep):
        if "sent_id" in filep[i]:
            sent=str(filep[i]).split("=")
            sent_id=sent[1].strip()
            i+=3
            flag=1
            continue
        if filep[i]=="\n":
            flag=0
        if flag==1:
            fragments=filep[i].split("\t")
            words_tag[fragments[1]].append(fragments[3])
            words_tag_short[fragments[1]].append(fragments[4])
            words_tag_position[fragments[1]].append(sent_id)
            if str(fragments[3]) in tag_count.keys():
               tag_count[str(fragments[3])]+=1
            else:
               tag_count[str(fragments[3])]=1 
            if str(fragments[4]) in tag_short_count.keys():
               tag_short_count[str(fragments[4])]+=1
            else:
               tag_short_count[str(fragments[4])]=1             
        i+=1
    
    return words_tag,words_tag_short,words_tag_position,tag_count,tag_short_count

words_tag_hin,words_tag_short_hin,words_tag_position_hin,tag_count_hin,tag_short_count_hin=counts("hindi.txt")

words_tag_fr,words_tag_short_fr,words_tag_position_fr,tag_count_fr,tag_short_count_fr=counts("french.txt")
tag_count_hin['_']=0


for key in tag_short_count_hin.keys():
    if key not in tag_short_count_fr.keys():
        tag_short_count_fr[key]=0

for key in tag_short_count_fr.keys():
    if key not in tag_short_count_hin.keys():
        tag_short_count_hin[key]=0

'''
x=np.arange(len(tag_short_count_fr))
plt.bar(x-0.4, tag_short_count_hin.values(),width=0.4,color='b',label='Hindi')
plt.bar(x, tag_short_count_fr.values(),width=0.4,color='g',label='French')
plt.legend(bbox_to_anchor=(0., 0.99, 1., .102), loc=3,
           ncol=2, mode="expand",fontsize=20, borderaxespad=0.)

plt.xticks(x,tag_short_count_hin.keys(),rotation=90)
plt.autoscale(tight=True)
plt.show()
'''

word_important_hindi={}
for hin in words_tag_position_hin:
    word_important_hindi[hin]=len(words_tag_position[hin])
    

sorted_word_important_hindi= sorted(word_important_hindi.iteritems(), key=operator.itemgetter(1))

word_important_fr={}
for fr in words_tag_position_fr:
    word_important_fr[fr]=len(words_tag_position[fr])
sorted_word_important_fr= sorted(word_important_fr.items(), key=operator.itemgetter(1))

h=[]
for i in range(len(sorted_word_important_hindi)-20,len(sorted_word_important_hindi)):
    h.append(sorted_word_important_hindi[i][1])

f=[]    
for i in range(len(sorted_word_important_fr)-20,len(sorted_word_important_fr)):
    f.append(sorted_word_important_fr[i][1])

'''
x=np.arange(20)
plt.bar(x-0.4,h,width=0.4,color='b',label='Hindi')
plt.bar(x,f,width=0.4,color='g',label='French')
plt.legend(bbox_to_anchor=(0., 0.99, 1., .102), loc=3,
           ncol=2, mode="expand",fontsize=20, borderaxespad=0.)

plt.autoscale(tight=True)
plt.show()
'''

hinkey=words_tag_position_hin.keys()
frkey=words_tag_position_fr.keys()

for hin in hinkey:
    if(len(words_tag_position_hin[hin])==1):
        words_tag_position_hin.pop(hin)

for fr in frkey:
    if(len(words_tag_position_fr[fr])==1):
        words_tag_position_fr.pop(fr)

word_assoc=[]
for hin in words_tag_position_hin.keys():
    for fr in words_tag_position_fr.keys():
        if (len(set(words_tag_position_hin[hin]).intersection(set(words_tag_position_fr[fr])))>3):
           word_assoc.append([hin,fr,len(set(words_tag_position_hin[hin]).intersection(set(words_tag_position_fr[fr])))**4/(1.0*(len(set(words_tag_position_hin[hin])))*(len(set(words_tag_position_hin[hin]))))])
       
sorted_word_assoc=sorted(word_assoc, key=operator.itemgetter(2),reverse=True)
print sorted_word_assoc[:20]

                    
