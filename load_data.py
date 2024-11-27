from email.mime import image
import numpy as np
import pandas as pd
import argparse
import random
from itertools import chain
import os
from datasets import load_dataset
from datasets import load_metric
from datasets import Dataset, Image
from sklearn.model_selection import KFold,train_test_split
from sklearn.utils import shuffle

def split_data(data_list,n_splits,dataset,key):
    if key==None:
        key_index=[i for i in range[data_list]]

    elif dataset=='CopCo':
        nond_indx,dyslexia_index=[],[]
        for index,d in enumerate(data_list):
            label = d['label']
            if label==0:
                nond_indx.append([index])
            else:
                dyslexia_index.append([index])
        max_no_samples = len(dyslexia_index)
        nond_indx = nond_indx[:max_no_samples]
        key_index = dyslexia_index+nond_indx
        key_index = shuffle(key_index)

    else:
        key_index_map={}
        for index,d in enumerate(data_list):
            k=d[key]
            if k not in key_index_map.keys():
                key_index_map[k]=[index]
            else:
                key_index_map[k].append(index)
        key_index=list(key_index_map.values())

    
    if dataset=='gazebase':
        train_index=np.array(key_index_map[1])
        val_index=np.array(key_index_map[2])
        test_index=np.array(key_index_map[2])
        train_val_test_index=[np.array([train_index,val_index,test_index])]
        return train_val_test_index

    train_val_test_index=[]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1799)
    for train_val_key_idx, test_key_idx in kf.split(key_index):
        if dataset=='zuco':
            train_index=[key_index[t] for t in train_val_key_idx]
            train_index=np.array(sum(train_index,[]))
            val_index=[key_index[t] for t in test_key_idx]
            val_index=np.array(sum(val_index,[]))
        elif dataset=='SB-SAT' or dataset=='CopCo':
            val_fold=n_splits-1
            val_cv = KFold(n_splits=val_fold, shuffle=True,random_state=1799,)
            for train_key_idx,val_key_idx in val_cv.split(train_val_key_idx):
                break
            train_index=[key_index[train_val_key_idx[t]] for t in train_key_idx]
            train_index=np.array(sum(train_index,[]))
            val_index=[key_index[train_val_key_idx[t]] for t in val_key_idx]
            val_index=np.array(sum(val_index,[]))
        test_index=[key_index[t] for t in test_key_idx]
        test_index=np.array(sum(test_index,[]))
        train_val_test_index.append(np.array([train_index,val_index,test_index]))
    
    return train_val_test_index

def load_image(Pdict_list, y, base_path, dataset_prefix):
    images_path = []
    labels = []
    texts = []
    scanpaths = []
    pids = []
    for idx, d in enumerate(Pdict_list):
        pid = d['id']
        text = d['text']
        scanpath = d["scanpath"]

        label = y[idx]
        labels.append(label)        
        texts.append(text)
        scanpaths.append(scanpath)
        image_path = os.path.join(base_path,f'{dataset_prefix}_images',f'{pid}.png')
        images_path.append(image_path)
        pids.append([int(d['subject_id']),d['sample_id']])

    datadict = {"image": images_path, "text": texts, 'scanpath':scanpaths, "label": labels, "sample_id":pids}
    dataset = Dataset.from_dict(datadict).cast_column("image", Image())
    
    return dataset, datadict

def get_data_split(base_path, Pdict_list, idx_train, idx_test, task, dataset, data_task, binary,upsample=False, prefix='', idx_val=[]):
    y=[]
    label_binary_map={0:0,1:0,2:1}
    for d in Pdict_list:
        if binary:
            y.append(label_binary_map[d['label']])
        else:
            if isinstance(d['label'],list):
                one_hot_label=[1 if i in d['label'] else 0 for i in range(11)]
                y.append(one_hot_label)
            elif data_task=='general comprehension':
                y.append(d['label']['overall_compre'])
            elif data_task=='comprehension':
                y.append(d['label']['compre'])
            elif data_task=='difficulty':
                y.append(d['label']['difficult'])    
            elif data_task=='native':
                y.append(d['label']['native']) 
            elif dataset=='CFILT-Sentiment_Analysis_II':
                if data_task=='sentiment analysis':
                    y.append(d['label']['sentiment'])
                elif data_task=='sarcasm detection':
                    y.append(d['label']['sarcasm'])
            else:
                y.append(d['label'])
    y=np.array(y) if isinstance(d['label'],list) else np.array(y).reshape((-1, 1))
    if task == "classification":
        y = y.astype(np.float32) if isinstance(d['label'],list) else y.astype(np.int32)
    elif task == "regression":
        y = y.astype(np.float32)

    # extract train/val/test examples
    # if pval is none: use test dataset instead
    if len(idx_val)==0:
        idx_val=idx_test
        print("Don't have val dataset, use test dataset as eval dataset instead")
    Ptrain = Pdict_list[idx_train]
    Pval = Pdict_list[idx_val]
    Ptest = Pdict_list[idx_test]
    
    ytrain = y[idx_train]
    yval = y[idx_val]
    ytest = y[idx_test]  

    # upsampling the training dataset
    if upsample:
        ytrain = y[idx_train]
        idx_0 = np.where(ytrain == 3)[0]
        idx_1 = np.where(ytrain == 1)[0]
        n0, n1 = len(idx_0), len(idx_1)
        print(n0, n1)
        if n0 > n1:
            idx_1 = random.choices(idx_1, k=n0)            
        else:
            idx_0 = random.choices(idx_0, k=n1)
        # make sure positive and negative samples are placed next to each other
        random.shuffle(idx_0)
        random.shuffle(idx_1)
        upsampled_train_idx = list(chain.from_iterable(zip(idx_0, idx_1)))
        Ptrain = Ptrain[upsampled_train_idx]
        ytrain = ytrain[upsampled_train_idx]
    
    # only remove part of params in val, test set
    train_dataset, train_datadict = load_image(Ptrain, ytrain, base_path, prefix)
    val_dataset, val_datadict = load_image(Pval, yval, base_path, prefix)
    test_dataset, test_datadict = load_image(Ptest, ytest, base_path, prefix)

    return train_dataset, val_dataset, test_dataset, ytrain, yval, ytest

