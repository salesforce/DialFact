
from ipywidgets import FloatProgress
import argparse
import torch
import jsonlines
import random
import os
import json
import numpy as np
from scipy.special import softmax

# os.environ["NCCL_SHM_DISABLE"] = "1"

from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, default_data_collator, set_seed
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score
import pickle
from os import path


def save_wiki_pickle(wiki_map, pathp='./'):
    if path.exists(pathp + 'wiki_map.pickle'):
        old_wiki_map = get_wiki_pickle()
        for k, v in old_wiki_map.items():
            if k not in wiki_map: 
                wiki_map[k] = v
    with open(pathp + 'wiki_map.pickle', 'wb') as handle:
        pickle.dump(wiki_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print('saved pickle wiki')
        
    return wiki_map

def get_wiki_pickle(pathp='./'):
    if path.exists(pathp + 'wiki_map.pickle'):
#         print('loading pickle from ', pathp + 'wiki_map.pickle')
        with open(pathp + 'wiki_map.pickle', 'rb') as handle:
            wiki_map = pickle.load(handle)
    else:
        print('creating file ', pathp + 'wiki_map.pickle')
        wiki_map = dict() 

    return wiki_map