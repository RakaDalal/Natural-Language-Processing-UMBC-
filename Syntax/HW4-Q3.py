#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 18:09:04 2017

@author: rakadalal
"""
import codecs
from collections import defaultdict 
import numpy as np
import sys

t=int(sys.argv[1])
root=sys.argv[2]
N=int(sys.argv[3])
path=sys.argv[4]
#terminals=["Chris", "eat", "eats", "sandwich", "a", "an", "apple", "Girls", "adopted", "beautiful", "big", "white", "dog", ".", "!", "?","","to","of","apples", "ugly", "enormous", "orange"]
#print len(terminals)
def create_terminals(path):
    terminals=[]
    keys=[]
    filepointer=codecs.open(path,'r',encoding='utf-8')
    filep=filepointer.readlines()
    for line in filep:
        line=line.split(" ")
        keys.append(line[1].strip())
        for i in range(3, len(line)):
            terminals.append(line[i].strip())
    terminals=set(terminals)
    terminals=list(terminals)
    for item in keys:
        if item in terminals:
            terminals.remove(item)
    terminals.append("")
    return (terminals)

def create_dic(terminals, path):
    wcfg=defaultdict(list)
    filepointer=codecs.open(path,'r',encoding='utf-8')
    filep=filepointer.readlines()
    for line in filep:
        line=line.split(" ")
        weight=float(line[0])
        key=line[1]
        temp_list=[]
        for i in range(3, len(line)):
            if line[i].strip() in terminals:
                tup=(line[i].strip(),"Terminal")
            else:
                tup=(line[i].strip(),"Non-Terminal")
            temp_list.append(tup)
        temp_list.append(weight)
        wcfg[key].append(temp_list)
        
    return wcfg
        
terminals=create_terminals(path)        
wcfg=create_dic(terminals, path)
curr=[]

for k in range(N):
    sen=[root]
    if (t==1):
        print sen
    while (True):
        flag=0
        for tag in sen:
            if tag not in terminals:
                flag=1
                non_terminal=tag
                #print non_terminal
                production=np.random.choice(np.arange(0,len(wcfg[non_terminal])), p=[wcfg[non_terminal][i][len(wcfg[non_terminal][i])-1] for i in range(len(wcfg[non_terminal]))]/np.sum([wcfg[non_terminal][i][len(wcfg[non_terminal][i])-1] for i in range(len(wcfg[non_terminal]))]))
                replace=[]
                for i in range(len(wcfg[non_terminal][production])-1):
                    replace.append(wcfg[non_terminal][production][i][0])
                sen=sen[:sen.index(tag)]+replace+sen[sen.index(tag)+1:]
                break
        if (t==1):
           print sen 
           print "\n"
        if(flag==0):
            break
    print ' '.join([str(x) for x in sen]) 
    print "\n"       


