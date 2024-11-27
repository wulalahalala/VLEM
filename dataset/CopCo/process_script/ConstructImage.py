import os
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from transformers import BertTokenizerFast
import statistics
import math


"""
Dataset configurations 
"""
typesetting = {
    "max_x" : 1920,
    "max_y" : 1080,
    "width" : 920,
    "start_x" : 127,
    "start_y" : 80,
    "rowledge" : 58,
    "word_size" : 32
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
    "orange": "3",
    "gold": "4",
}
tokenizer=BertTokenizerFast.from_pretrained('../../../models/bert/bert-base-multilingual-cased')

def get_text_token_map(texts):
    if os.path.exists('../raw_data/stimuli/token_map.csv'):
        df_map=pd.read_csv('../raw_data/stimuli/token_map.csv')
    else:
        df_map=pd.DataFrame(columns=['page_name','original','converted','word','line'])
        for text in texts:
            if text=='\n':
                continue
            text=text.strip()
            page_name,line,text=text.split('>')
            add_num = 4
            words=text.split(' ')
            word_count=0
            for original_id,w in enumerate(words):
                original_id = original_id+add_num
                tokens=tokenizer(w,add_special_tokens=False)
                word_ids = list(set(tokens.word_ids()))
                converted_id = [i+word_count for i in word_ids]
                word_count+=len(word_ids)
                df_map=df_map.append({'page_name':page_name,'original':original_id,'converted':converted_id,'word':w,'line':int(line)},ignore_index=True)
        df_map.to_csv('../raw_data/stimuli/token_map.csv',index=False)
    return df_map

def get_data_label(data,subject_id,labels,word_info):
    data_df = pd.read_csv(data,sep='\t')
    data_df['speechid'] = data_df['speechid'].astype(str)
    data_df.drop(data_df[data_df['speechid'].str.contains('UNDEFINEDnull')].index, inplace=True)
    data_df['speechid'] = data_df['speechid'].astype(int)
    data_df['TRIAL_INDEX'] = data_df['TRIAL_INDEX'].astype(int)
    word_df = pd.read_csv(word_info)
    word_df["uniqueID"] = word_df["speechId"].astype(str) + "-" + word_df["trialId"].astype(str)
    grouped_word = word_df.groupby('uniqueID')
    data_list = []
    row_scale = []
    sample_index = 0
    for uniqueID,sample_word in grouped_word:
        #read sample
        sample_df=data_df[(data_df['speechid']==int(uniqueID.split('-')[0]))&(data_df['TRIAL_INDEX']==int(uniqueID.split('-')[1]))].reset_index(drop=True)
        words=sample_word['word'].tolist()
        text=' '.join(words)
        word_token_ids = []
        accumulated_token_num = 0
        for wi,wrow in sample_word.iterrows():
            tokens=tokenizer(wrow['word'],add_special_tokens=False)
            word_ids = list(set(tokens.word_ids()))
            converted_id = [i + accumulated_token_num for i in word_ids]
            accumulated_token_num += len(word_ids)
            word_token_ids.append(converted_id)
        sample_word['converted_ids']=word_token_ids

        time, x, y, row_list, rowledge= [], [], [], [], []
        scanpath_tokens, scanpath_token_ids, scanpath_time =[], [], []
        for index,row in sample_df.iterrows():
            fix_x=int(row['CURRENT_FIX_X'].split(',')[0])
            fix_y=int(row['CURRENT_FIX_Y'].split(',')[0])
            fix_duration=row['CURRENT_FIX_DURATION']
            past_time = 0 if index == 0 else time[-1] + 1
            fix_time = [i + past_time for i in range(fix_duration)]
            time += fix_time
            x += [fix_x] * fix_duration
            y += [fix_y] * fix_duration

            fix_char_id = row['CURRENT_FIX_INTEREST_AREA_INDEX']
            if fix_char_id!='.':
                word_row = sample_word[sample_word['char_IA_ids'].apply(lambda x:fix_char_id in x)]
                if word_row.shape[0]!=0:
                    token_id = word_row['converted_ids'].values[0]
                    scanpath_token_ids += token_id
                    scanpath_tokens += [word_row['word'].values[0]] * len(token_id)
                    scanpath_time += [time[-1]] * len(token_id)

                    fix_area = row['CURRENT_FIX_INTEREST_AREA_DATA'].split(',')
                    top = float(fix_area[-3])
                    bottom = float(fix_area[-1][:-1])
                    row_list+=[top,bottom]
                    rowledge.append(bottom-top)

            next_sac_duration=row['NEXT_SAC_DURATION']
            if next_sac_duration!='.' and index!=sample_df.shape[0]-1:
                time += [time[-1]+int(next_sac_duration)]
                x += [int(sample_df.loc[index+1,'CURRENT_FIX_X'].split(',')[0])]
                y += [int(sample_df.loc[index+1,'CURRENT_FIX_Y'].split(',')[0])]

        if len(rowledge)==0:
            continue
        rowledge=statistics.mode(rowledge)
        row_scale.append([min(row_list), max(row_list), rowledge])
        sample_data={'subject':subject_id,
                     'time':time,
                     'x':x,
                     'y':y,
                     'scan_tokens':scanpath_tokens,
                     'scan_token_ids':scanpath_token_ids,
                     'scan_time':scanpath_time,
                     'label':labels,
                     'sample':sample_index,
                     'text':text}
        data_list.append(sample_data)
        sample_index+=1


    return data_list, row_scale


def draw_image(pid, max_time, times, xs, ys, row_scale,
                override,image_size, grid_layout,
                linestyle, linewidth,
                color_mapping,min_duration):
    # set matplotlib param
    grid_height = grid_layout[0]
    grid_width = grid_layout[1]
    if image_size is None:
        cell_height = 224/2
        cell_width = 224
        img_height = grid_height * cell_height
        img_width = grid_width * cell_width
    else:
        img_height = image_size[0]
        img_width = image_size[1]

    dpi = 100
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

    ts_time=np.array(times).reshape(-1, 1).astype('float64')
    plt.rcParams['xtick.direction'] = 'in'
    x_ticks = np.arange(min_duration, max_time, min_duration)
    #horizontal positions
    x_value = np.array(xs).reshape(-1, 1).astype('float64')
    param_color = color_mapping['x']
    plt.subplot(grid_height, grid_width, 1)
    plt.plot(ts_time, x_value, linestyle=linestyle, linewidth=linewidth, color=param_color)
    plt.xlim(0,max_time)
    plt.ylim(0,typesetting['max_x'])
    # plt.xticks([])
    plt.xticks(x_ticks, [''] * len(x_ticks))
    plt.yticks([])

    #lengthwise position
    y_value = np.array(ys).reshape(-1, 1).astype('float64')
    param_color = color_mapping['y']
    plt.subplot(grid_height, grid_width, 2)
    # line split bar for the text area
    bar_num = math.ceil((row_scale[1]-row_scale[0])/row_scale[2])
    strat_y = row_scale[0]
    for k in range(bar_num):
        color_id=k%2
        plt.axhspan(ymin=strat_y,ymax=strat_y+row_scale[2],color=color_mapping[f'split_{color_id}'],alpha=0.5)
        strat_y+=row_scale[2]
    plt.plot(ts_time, y_value, linestyle=linestyle, linewidth=linewidth, color=param_color)
    plt.xlim(0,max_time)
    plt.ylim(0,typesetting['max_y'])
    # plt.xticks([])
    plt.xticks(x_ticks, [''] * len(x_ticks))
    plt.yticks([])


    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(img_path, pad_inches=0)
    plt.clf()

    return None


def construct_image(
        data_path=None,
        subject_id=None,
        all_label=None,
        word_info=None,
        linestyle="-", linewidth=1,
        override=False,
        outlier=None,
        interpolation=False,
        image_size=None,
        grid_layout=(3, 1),
        patch_num=16
):
    # load data
    data, row_scales = get_data_label(data_path,subject_id,all_label,word_info)

    duration_list= []
    for d in data:
        duration_list.append(list(d['time'])[-1])
        max_time = max(duration_list)
    min_duration = min(duration_list)

    # construct the mapping from param to marker, color, and idx
    num_samples = len(data)
    params = list(data[0].keys())
    plt_colors = list(color_detailed_description.keys())
    idx_mapping = {}
    color_mapping = {}
    for idx, param in enumerate(params[2:4] + ['split_0', 'split_1']):
        color_mapping[param] = plt_colors[idx]
        idx_mapping[param] = idx

    with open('../processed_data/param_idx_mapping.json', 'w') as f:
        json.dump(idx_mapping, f)
    with open('../processed_data/param_color_mapping.json', 'w') as f:
        json.dump(color_mapping, f)

    scanpath_labels=[]
    global sample_num
    # draw the image for each sample
    for idx, p in enumerate(data):
        pid = sample_num+idx
        sample_id = p['sample']
        times = p['time']
        xs = p['x']
        ys = p['y']
        label = p['label']
        subject_id = p['subject']
        row_scale = row_scales[idx]

        # split scanpath with patch
        scanpath = p['scan_token_ids']
        if len(scanpath)==0:
            continue
        scanpath_times = p['scan_time']
        text = p['text']
        patch_row_num = int(image_size[1] / patch_num)
        split_scan = [[] for i in range(patch_row_num)]
        # Adaptive time range
        max_time = int(max(times))
        patch_time = max_time / patch_row_num
        for scan_id, scan_time in enumerate(scanpath_times):
            split_id = int(scan_time / patch_time)
            if split_id==patch_row_num:
                split_scan[-1].append(scanpath[scan_id])
            else:
                split_scan[split_id].append(scanpath[scan_id])


        # draw the image for each p
        draw_image(pid, max_time, times, xs, ys, row_scale,
                                   override,image_size, grid_layout,
                                   linestyle, linewidth,
                                   color_mapping,min_duration)


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
    word_path = os.path.join(root_path,'ExtractedFeatures')
    data_path = os.path.join(root_path,'FixationReports')
    nond_subjects = ["02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "15", "16", "18", "19", "20",
                     "21", "22"]
    dys_subjects = ["23", "24", "25", "26", "27", "28", "29", "30", "31", "33", "34", "35", "36", "37", "38", "39",
                    "40", "41"]
    subject_ids ={'non':nond_subjects,'dys':dys_subjects}

    scanpath_labels=[]
    sample_num=0
    if not os.path.exists('../processed_data'): os.mkdir('../processed_data')
    for subject_type,subject_list in subject_ids.items():
        label=0 if subject_type=='non' else 1
        for sub_id in tqdm(subject_list):
            sub_data_path = os.path.join(data_path,f"FIX_report_P{sub_id}.txt")
            sub_word_path = os.path.join(word_path,f"P{sub_id}.csv")
            scanpath_label=construct_image(
                data_path=sub_data_path,
                subject_id=sub_id,
                all_label=label,
                word_info=sub_word_path,
                linestyle="-", linewidth=1,
                override=True,
                outlier=None,
                interpolation=False,
                image_size=(224,224),
                grid_layout=(2, 1),
                patch_num=16
            )
            scanpath_labels+=scanpath_label
    np.save('../processed_data/duration/Scanpath.npy',scanpath_labels)











