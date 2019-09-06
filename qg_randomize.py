#SQuADのデータ処理
#必要条件:CoreNLP
#Tools/core...で
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

import os
import sys
sys.path.append("../")
import json
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize,sent_tokenize
import pickle
import collections
import random
from nltk.corpus import stopwords
from nltk.corpus import stopwords

sentences=[]
questions=[]

random.seed(0)

setting="-du"

with open("data/squad-src-train-du.txt")as f:
    for line in f:
        sentences.append(line.strip())

with open("data/squad-tgt-train-du.txt")as f:
    for line in f:
        questions.append(line.strip())

random_list=list(range(len(questions)))
random.shuffle(random_list)
with open("data/squad-src-train{}.txt".format(setting),"w")as f:
    for i in random_list:
        f.write(sentences[i]+"\n")
with open("data/squad-tgt-train{}.txt".format(setting),"w")as f:
    for i in random_list:
        f.write(questions[i]+"\n")
