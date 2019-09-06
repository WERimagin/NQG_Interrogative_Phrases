import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from onmt.utils.corenlp import CoreNLP

from tqdm import tqdm
from collections import defaultdict
import collections
import math
from statistics import mean, median,variance,stdev
import random
import json
import argparse
import os.path
import numpy as np





parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="data/squad-src-val-interro.txt", help="input model epoch")
parser.add_argument("--tgt", type=str, default="data/squad-tgt-val-interro.txt", help="input model epoch")
parser.add_argument("--pred", type=str, default="data/squad-tgt-val-interro.txt", help="input model epoch")
parser.add_argument("--interro", type=str, default="data/squad-interro-val-interro.txt", help="input model epoch")
parser.add_argument("--noninterro", type=str, default="data/squad-noninterro-val-interro.txt", help="input model epoch")
parser.add_argument("--p_noninterro", type=str, default="data/squad-pred_noninterro-val-interro.txt", help="input model epoch")

parser.add_argument("--tgt_interro", type=str, default="", help="input model epoch")
parser.add_argument("--not_interro", action="store_true")
parser.add_argument("--same_interro", action="store_true")
parser.add_argument("--each_interro", action="store_true")

parser.add_argument("--ratio", type=float, default=1.0, help="input model epoch")
parser.add_argument("--print", type=int ,default=10)
parser.add_argument("--type", type=int ,default=0)
args = parser.parse_args()

random.seed(0)

srcs=[]
targets=[]
predicts=[]
interros=[]
t_noninterros=[]
p_noninterros=[]

with open(args.src,"r")as f:
    for line in f:
        srcs.append(line.strip())

with open(args.pred,"r")as f:
    for line in f:
        predicts.append(line.strip())

with open(args.interro,"r")as f:
    for line in f:
        interros.append(line.strip())


interro_list=["what","who","when","where","how","why","which", \
        "in what year","how many","how much","how long","how many times", \
        "what year","how old"]

if args.type==0:
    id=0
    for i in range(0,len(interro_list)*args.print,len(interro_list)):
        print(srcs[i])
        for j,interro in enumerate(interro_list):
            print(predicts[id])
            id+=1
        print()

if args.type==1:
    srcs_set=list(set(srcs))
    new_srcs=[]
    np.random.seed(0)
    random_list=np.random.permutation(np.arange(len(srcs_set)))
    for i in random_list[0:300]:
        for interro in interro_list:
            new_srcs.append(" ".join([srcs_set[i],"<SEP>",interro]))
    print("len:{}".format(len(new_srcs)))
    print("len interrolist:{}".format(len(interro_list)))
    with open("data/squad-src-test-appendinterro.txt","w")as f:
        for s in new_srcs:
            f.write(s+"\n")



if args.type==2:
    onedict=defaultdict(int)
    twodict=defaultdict(int)
    alldict=defaultdict(int)

    for i in range(len(interros)):
        interro=interros[i].split()
        if len(interro)>=1:
            onedict[interro[0]]+=1
        if len(interro)>=2:
            twodict[" ".join([interro[0],interro[1]])]+=1
        alldict[" ".join(interro)]+=1

    onedict=sorted(onedict.items(),key=lambda x: -x[1])
    twodict=sorted(twodict.items(),key=lambda x: -x[1])
    alldict=sorted(alldict.items(),key=lambda x: -x[1])

    for key,value in onedict[0:30]:
        print(key,value)
    print()
    for key,value in twodict[0:30]:
        print(key,value)
    print()
    for key,value in alldict[0:30]:
        print(key,value)
