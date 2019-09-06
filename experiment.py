import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from tqdm import tqdm
from collections import defaultdict,Counter
import collections
import math
from statistics import mean, median,variance,stdev
import random
import json
import argparse
import numpy as np




parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="data/squad-src-val-interro.txt", help="input model epoch")
parser.add_argument("--tgt", type=str, default="data/squad-tgt-val-interro.txt", help="input model epoch")
parser.add_argument("--pred", type=str, default="pred.txt", help="input model epoch")
parser.add_argument("--interro", type=str, default="data/squad-interro-val-interro.txt", help="input model epoch")

parser.add_argument("--tgt_interro", type=str, default="", help="input model epoch")
parser.add_argument("--not_interro", action="store_true")
parser.add_argument("--all_interro", action="store_true")
parser.add_argument("--interro_each", action="store_true")

parser.add_argument("--notsplit", action="store_true")
parser.add_argument("--print", action="store_true")
parser.add_argument("--result", action="store_true")




args = parser.parse_args()

interros=[]


with open(args.interro,"r")as f:
    for line in f:
        interros.append(line.rstrip())

interros_num=[len(i.split()) for i in interros]
c_dict=Counter(interros_num)
for k,v in c_dict.items():
    print(k,v/len(interros_num))
