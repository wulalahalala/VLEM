import math
import os
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from transformers import BertTokenizerFast
import scipy.io as io
import collections
import re
from matplotlib.ticker import NullLocator

import warnings
warnings.filterwarnings("ignore")


"""
Dataset configurations 
"""
typesetting = {
    "max_x" : 800,
    "max_y" : 600,
    "rowledge" : 40,
}

param_detailed_description = {
    "Time": "Time",
    "X": "X-coordinate",
    "Y": "Y-coordinate",
    "Pupil": "Pupil diameter",
}

color_detailed_description = {
    "red": "1",
    "green": "2",
    "blue": "3",
    "orange": "4",
    "gold": "5",
}

tokenizer=BertTokenizerFast.from_pretrained('../../../models/bert/bert-base-uncased')

def retrieve_label(sentence_label,label_list_path,task_type):
    label_list=next(os.walk(label_list_path))[1]
    if task_type == 'multiclass':
    # multicalss
        df = pd.DataFrame(columns=['original_sentence', 'split_sentence', 'label','word_start','word_end'])
        for label_id,label_type in enumerate(label_list):
            type_file = next(os.walk(os.path.join(label_list_path,label_type)))[2]
            for file in type_file:
                sentence_index=int(file.split("_")[0])
                original_sentence=sentence_label.loc[sentence_index,'sentence']

                with open(os.path.join(label_list_path,label_type,file),'r',encoding='utf-8') as f:
                    split_sentence=f.readline()
                split_sentence=split_sentence.replace('<e> ','',1).replace(' </e> ','',1).replace(' <e> ','',1).replace(' </e>','',1).strip()

                original_words = original_sentence.split(' ')
                split_words = split_sentence.split(' ')
                s=e=-1
                for o_idx in range(len(original_words)):
                    if o_idx>len(original_words)-3:
                        break
                    if original_words[o_idx:o_idx+2]==split_words[:2]:
                        s=o_idx
                    if original_words[o_idx:o_idx + 2] == split_words[-2:]:
                        e = o_idx+2
                s=e-len(split_words) if s==-1 and e!=-1 else s
                e = s + len(split_words) if s != -1 and e == -1 else e
                data_to_insert = {'original_sentence': original_sentence,'split_sentence': split_sentence,'label': label_id,'word_start':s,'word_end':e}
                df = df.append(data_to_insert,ignore_index=True)
        return df

    if task_type=='multilabel':
        sentence_label['label'] = pd.Series([[] for i in range(len(sentence_label))] )
        for label_index, label_name in enumerate(label_list):
            type_list = next(os.walk(os.path.join(label_list_path,label_name)))[2]
            for type_name in type_list:
                row_id = int(type_name.split('_')[0])
                if label_index not in sentence_label.loc[row_id,'label']:
                    sentence_label.loc[row_id,'label'].append(label_index)
        return sentence_label


def get_data_label(results_file,raw_data_list,subject_id,sentence_label,sentence):

    data = io.loadmat(results_file, squeeze_me=True, struct_as_record=False,simplify_cells=True)['sentenceData']
    row_scale,data_list = [],[]

    for sample_id,d in enumerate(tqdm(data)):
        text = d['content']
        if 'task1' in results_file:
            sentence_id = int(sentence.loc[sentence['sentence']==text,'ID'])
            try:
                label = int(sentence_label.loc[sentence_label['sentence_id'] == str(sentence_id), 'sentiment_label'])
            except Exception as e:
                label = int(sentence_label.loc[sentence_label['sentence_id'] == str(sentence_id), 'control'])
        else:
            sentence_id = sentence.loc[sentence['sentence']==text,['paragraph_id','sentence_id']]
            sentence_id.reset_index(drop=True, inplace=True)
            try:
                label = sentence_label.loc[(sentence_label['paragraph_id'] == sentence_id['sentence_id'][0])&(sentence_label['sentence_id'] == sentence_id['paragraph_id'][0]), 'label']
                label.reset_index(drop=True, inplace=True)
                label = label[0]
                if len(label)==0:
                    continue
            except Exception as e:
                continue

        if isinstance(d['allFixations'],dict):
            if type(d['allFixations']['x']) != np.ndarray:
                continue
            elif len(d['allFixations']['x'])==0:
                continue
        elif np.isnan(d['allFixations']):
            continue


        sample_row_scle = []
        sample_data = {'subject': subject_id}
        info_list,row_list = [],[]
        #1.sample time range
        start_time,end_time = 99999999999999,0
        d_word = d['word']
        word_add_num=0
        for w_id,w in enumerate(d_word):
            row_list+=[d['wordbounds'][w_id,1],d['wordbounds'][w_id,3]]
            w_content=w['content']
            word_ids=list(set(tokenizer(w_content,add_special_tokens=False).word_ids()))
            word_ids_len=len(word_ids)
            info_list.append([w['content'],d['wordbounds'][w_id,:],w_id+word_add_num,word_ids_len])
            word_add_num=word_add_num+word_ids_len-1
            try:
                w_rawET = w['rawET']
            except Exception as e:
                continue
            if len(w_rawET)!=0:
                if w['nFixations']==1:
                    start_time = w_rawET[0, 0] if w_rawET[0, 0] < start_time else start_time
                    end_time = w_rawET[0, -1] if w_rawET[0, -1] > end_time else end_time
                else:
                    for w_raw_data in w_rawET:
                        if len(w_raw_data)!=0:
                            start_time = w_raw_data[0,0] if w_raw_data[0,0]<start_time else start_time
                            end_time = w_raw_data[0,-1] if w_raw_data[0,-1]>end_time else end_time

        #2.compare with raw data to retrieve corresponding raw data
        for raw_data in raw_data_list:
            time_range=raw_data['time_range']
            if start_time>time_range[0] and end_time<time_range[1]:
                all_raw_data = raw_data['data']
                rd = all_raw_data[(all_raw_data[:,0]>=start_time) & (all_raw_data[:,0]<=end_time)]
                all_fixation = raw_data['fixations']
                fixation_info = all_fixation[(all_fixation[:,1]>=start_time) & (all_fixation[:,1]<=end_time)]
                break
        rd[np.where((rd[:,1]==0)&(rd[:,2]==0))[0]] = -1
        sort_time = sorted(list(set(rd[:,0])))
        try:
            start_time =sort_time[0] if sort_time[0]!=-1 else sort_time[1]
        except Exception as e:
            continue
        time = [i - start_time if i!=-1 else -1 for i in rd[:,0]]


        #3.scanpath
        f_i = fixation_info.copy()
        for fi_row_index in range(f_i.shape[0]):
            for f_row_index in range(d['allFixations']['pupilsize'].shape[0]):
                if (d['allFixations']['pupilsize'][f_row_index]==f_i[fi_row_index,5]) & (d['allFixations']['x'][f_row_index]==f_i[fi_row_index,3]):
                    fixation_info[fi_row_index,4] = d['allFixations']['y'][f_row_index]
                    break

        scanpath_token_ids, scanpath_tokens, scanpath_time = [], [], []
        for fixation in fixation_info:
            for word_info in info_list:
                if word_info[1][0]<=fixation[3]<=word_info[1][2] and word_info[1][1]<=fixation[4]<=word_info[1][3]:
                    scanpath_token_ids+=[word_info[2]+i for i in range(word_info[3])]
                    scanpath_tokens+=[word_info[0]]*word_info[3]
                    scanpath_time+=[fixation[1]]*word_info[3]
                    break
        scanpath_time = [i-start_time for i in scanpath_time]

        #4.stimulation layout information
        word_y_count=len(collections.Counter(d['allFixations']['y']))
        rowledge=(max(d['allFixations']['y'])-min(d['allFixations']['y']))/(word_y_count-1) if word_y_count!=1 else 60
        sample_row_scle += [min(row_list), max(row_list), word_y_count, rowledge]


        # tokens = tokenizer(text, add_special_tokens=False)
        # words = tokenizer.tokenize(text)
        # word_ids = list(set(tokens.word_ids()))
        sample_data['time'] = time
        sample_data['x'] = rd[:, 1]
        sample_data['y'] = rd[:, 2]
        sample_data['pupil'] = rd[:, 3]
        sample_data['scan_tokens']=scanpath_tokens
        sample_data['scan_token_ids'] = scanpath_token_ids
        sample_data['scan_time'] = scanpath_time
        sample_data['label'] = label
        sample_data['sample'] = sample_id
        sample_data['corpus'] = text

        data_list.append(sample_data)
        row_scale.append(sample_row_scle)



    return data_list, row_scale


def draw_image(pid, max_time, times, xs, ys, pupils, label, pupil_scale, row_scale,
                override,image_size, grid_layout,
                linestyle, linewidth,
                color_mapping, idx_mapping, min_duration):
    # set matplotlib param
    grid_height = grid_layout[0]
    grid_width = grid_layout[1]
    if image_size is None:
        cell_height = 224/3
        cell_width = 224
        img_height = grid_height * cell_height
        img_width = grid_width * cell_width
    else:
        img_height = image_size[0]
        img_width = image_size[1]

    dpi = 500
    plt.rcParams['savefig.dpi'] = dpi  # default=100
    plt.rcParams['figure.figsize'] = (img_width / dpi, img_height / dpi)
    plt.rcParams['figure.frameon'] = False

    # save path
    base_path = f"duration_{grid_height}_{grid_width}_{int(img_height)}_{img_width}_{linewidth}_images"
    base_path = "../processed_data/duration/" + base_path

    if not os.path.exists(base_path): os.makedirs(base_path)
    img_path = os.path.join(base_path, f"{pid}.png")
    if os.path.exists(img_path):
        if not override:
            return []

    #deal with blinking
    min_blink_time,max_blink_time,blink_gradient=25,100,20
    times=[-2 for i in range(max_blink_time)]+times+[-2 for i in range(max_blink_time)]
    times_without_blink = times.copy()
    pupil_gradient=[0,0]+[abs(pupils[i]-pupils[i-2]) for i in range(2,len(pupils))]
    for index,time in enumerate(times[max_blink_time:-max_blink_time]):
        times_index=index+max_blink_time
        if time==-1:
            times_without_blink[times_index-min_blink_time:times_index+min_blink_time]=[-1 for k in range(min_blink_time*2)]
        # elif pupil_gradient[index]>=blink_gradient and -1 in times[times_index-max_blink_time:times_index+max_blink_time]:
        #     times_without_blink[times_index]=-1
    times_without_blink=times_without_blink[max_blink_time:-max_blink_time]

    #Unified abscissa to maximum time and handling missing value
    max_length = int(max_time / 2) #sampling rate of 500 Hz
    ts_time = [-1 for i in range(max_length)]
    ts_time[:len(times_without_blink)] = times_without_blink
    ts_time = np.array(ts_time).reshape(-1, 1).astype('float64')
    miss_index = ts_time==-1
    ts_time[miss_index] = np.nan

    plt.rcParams['xtick.direction'] = 'in'
    x_ticks=np.arange(min_duration,max_time,min_duration)
    #horizontal positions
    x_value = [-1 for i in range(max_length)]
    x_value[:len(xs)]=xs
    x_value = np.array(x_value).reshape(-1, 1).astype('float64')
    x_value[miss_index] = np.nan
    param_color = color_mapping['x']
    plt.subplot(grid_height, grid_width, 1)
    plt.plot(ts_time, x_value, linestyle=linestyle, linewidth=linewidth, color=param_color)
    plt.xlim(0,max_time)
    plt.ylim(0,typesetting['max_x'])
    # plt.xticks([])
    plt.xticks(x_ticks,[''] * len(x_ticks))
    plt.yticks([])

    #lengthwise position
    y_value = [-1 for i in range(max_length)]
    y_value[:len(ys)] = ys
    y_value = np.array(y_value).reshape(-1, 1).astype('float64')
    y_value[miss_index] = np.nan
    param_color = color_mapping['y']
    plt.subplot(grid_height, grid_width, 2)
    # line split bar for the text area

    bar_num = row_scale[2]
    strat_y = row_scale[0]
    for k in range(bar_num):
        color_id=k%2
        plt.axhspan(ymin=strat_y,ymax=strat_y+row_scale[3],color=color_mapping[f'split_{color_id}'],alpha=0.5)
        strat_y+=row_scale[3]

    plt.plot(ts_time, y_value, linestyle=linestyle, linewidth=linewidth, color=param_color)
    plt.xlim(0,max_time)
    plt.ylim(0,typesetting['max_y'])
    # plt.xticks([])
    plt.xticks(x_ticks, [''] * len(x_ticks))
    plt.yticks([])

    #pupil
    p_value = [-1 for i in range(max_length)]
    p_value[:len(pupils)] = pupils
    p_value = np.array(p_value).reshape(-1, 1).astype('float64')
    p_value[miss_index] = np.nan
    param_color = color_mapping['pupil']
    plt.subplot(grid_height, grid_width, 3)
    plt.plot(ts_time, p_value, linestyle=linestyle, linewidth=linewidth, color=param_color)
    plt.xlim(0,max_time)
    plt.ylim(pupil_scale)
    plt.xticks(x_ticks, [''] * len(x_ticks))
    # plt.xticks([])
    plt.yticks([])

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(img_path, pad_inches=0)
    plt.clf()

    return None


def construct_image(
        raw_data_path=None,
        results_file=None,
        label_path=None,
        subject_id=None,
        linestyle="-", linewidth=1,
        task_type=None,
        override=False,
        outlier=None,
        interpolation=False,
        image_size=None,
        grid_layout=(3, 1),
        patch_num=16
):
    # get raw data
    raw_data_list = []
    raw_data_files = os.listdir(raw_data_path)
    raw_data_files = [i for i in raw_data_files if 'ET' in i]
    for raw_data_file in raw_data_files:
        raw_data_file_path = os.path.join(raw_data_path,raw_data_file)
        raw_data = io.loadmat(raw_data_file_path, squeeze_me=True, struct_as_record=False,simplify_cells=True)
        fixation_events = raw_data['eyeevent']['fixations']['data']
        raw_data_info = {"time_range":[raw_data['data'][0,0],raw_data['data'][-1,0]],'data':raw_data['data'],'fixations':fixation_events}
        raw_data_list.append(raw_data_info)

    # get sentence and label
    sentence_label = pd.read_csv(label_path,sep=';' if "1" in label_path else ',')
    sentence_path = os.path.join(raw_data_path,'..','sentence.csv')
    sentence = pd.read_csv(sentence_path,sep=';')
    if 'task2' in label_path:
        label_list_path=os.path.join(os.path.dirname(label_path),'zuco_nr_cleanphrases')
        sentence_label = retrieve_label(sentence_label,label_list_path,task_type)

    # load data
    subject_name = os.path.basename(raw_data_path)
    data, row_scales = get_data_label(results_file,raw_data_list,subject_name,sentence_label,sentence)

    duration_list,pupil_list,x,y = [],[],[],[]
    for d in data:
        duration_list.append(list(d['time'])[-1])
        pupil_list += list(d['pupil'])
    min_duration = min(duration_list)
    pupil_list = sorted(list(set(pupil_list)))
    min_pupil = pupil_list[0] if pupil_list[0]!=-1 else pupil_list[1]
    pupil_scale = [min_pupil,max(pupil_list)]

    params = list(data[0].keys())
    num_samples = len(data)
    # print(f"{num_samples} sample of {subject_id} with max time {max_time}!")
    # print(params)  # ['time', 'x', 'y', 'pupil']

    # construct the mapping from param to marker, color, and idx
    plt_colors = list(color_detailed_description.keys())
    idx_mapping = {}
    color_mapping = {}
    for idx, param in enumerate(params[2:5] + ['split_0', 'split_1']):
        color_mapping[param] = plt_colors[idx]
        idx_mapping[param] = idx

    with open('../processed_data/param_idx_mapping.json', 'w') as f:
        json.dump(idx_mapping, f)
    with open('../processed_data/param_color_mapping.json', 'w') as f:
        json.dump(color_mapping, f)

    scanpath_labels=[]
    global sample_num
    # draw the image for each sample
    for idx, p in enumerate(tqdm(data)):
        pid = sample_num+idx
        sample_id = p['sample']
        times = p['time']
        xs = p['x']
        ys = p['y']
        pupils = p['pupil']
        label = p['label']
        label = 2 if label==-1 else label
        row_scale = row_scales[idx]

        # split scanpath with patch
        scanpath = p['scan_token_ids']
        if len(scanpath)==0:
            continue
        scanpath_times = p['scan_time']
        text = p['corpus']
        patch_row_num = int(image_size[1] / patch_num)
        split_scan = [[] for i in range(patch_row_num)]
        #Adaptive time range
        max_time = int(max(times))
        patch_time = max_time / patch_row_num
        for scan_id, scan_time in enumerate(scanpath_times):
            split_id = int(scan_time / patch_time)
            split_id = patch_row_num-1 if split_id==patch_row_num else split_id
            split_scan[split_id].append(scanpath[scan_id])

        # draw the image for each p
        draw_image(pid, max_time, times, xs, ys, pupils, label, pupil_scale, row_scale,
                                   override,image_size, grid_layout,
                                   linestyle, linewidth,
                                   color_mapping, idx_mapping,min_duration)


        scanpath_label={'id':pid,
                        'text':text,
                        'scanpath':split_scan,
                        'label':label,
                        'subject_id':subject_id,
                        'sample_id':sample_id}
        scanpath_labels.append(scanpath_label)

    sample_num+=num_samples
    return scanpath_labels
if __name__ == "__main__":
    root_path = '../rawdata'
    task = 'task1- SR'   # task1- SR, task2- NR, task3- TSR
    task_type ='multiclass'
    label_map = {'task1- SR':'sentiment_labels_task1.csv','task2 - NR':'relations_labels_task2.csv','task3 - TSR':'relations_labels_task3.csv'}
    label_file = label_map[task]
    label_path = os.path.join(root_path,'task_materials',label_file)

    root_path = os.path.join(root_path,task)
    subject_list = next(os.walk(os.path.join(root_path,'Raw data')))[1]
    scanpath_labels=[]
    sample_num=0

    if not os.path.exists('../processed_data'): os.mkdir('../processed_data')
    for subject_id,subject in enumerate(tqdm(subject_list)):
        raw_data_path = os.path.join(root_path, 'Raw data',subject)
        results_file = os.path.join(root_path, 'Matlab files', f'results{subject}_{"SR" if "1" in task else "NR"}.mat')
        scanpath_label=construct_image(
            raw_data_path=raw_data_path,
            results_file = results_file,
            label_path = label_path,
            subject_id = subject_id,
            task_type = task_type,
            linestyle="-", linewidth=1,
            override=True,
            outlier=None,
            interpolation=False,
            image_size=(1120,1120),
            grid_layout=(3, 1),
            patch_num=16
        )
        scanpath_labels+=scanpath_label
    np.save('../processed_data/duration/Scanpath.npy',scanpath_labels)











