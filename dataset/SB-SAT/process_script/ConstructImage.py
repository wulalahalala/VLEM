import os
import pandas as pd
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from transformers import BertTokenizerFast


"""
Dataset configurations 
"""
typesetting = {
    "max_x" : 920,
    "max_y" : 690,
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
    "blue": "3",
    "orange": "4",
    "gold": "5",
}
tokenizer=BertTokenizerFast.from_pretrained('../../../models/bert/bert-base-uncased')

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

def get_data_label(data,subject_id,labels,token_id_map):
    global overall_median
    global passage_median
    subject_id=int(subject_id[3:])
    sample_list = data['page_name'].unique()
    sample_num = len(sample_list)
    data_list = [{'subject':subject_id} for i in range(sample_num)]
    row_scale = [[] for i in range(sample_num)]

    for sample_index,sample_id in enumerate(sample_list):
        #read sample
        sample_df=data[data['page_name']==sample_id].reset_index(drop=True)
        page_name=sample_df.loc[0,'page_name']
        book_name=sample_df.loc[0,'book_name']
        sample_token_map=token_id_map[token_id_map['page_name']==page_name].reset_index(drop=True)
        words=sample_token_map['word'].tolist()
        text=' '.join(words)
        row_list = sample_token_map.loc[0,'line']

        #get label

        book_name = sample_df.loc[0,'book_name']
        sample_labels = labels[labels['book']==book_name]
        overall_compre = sample_labels['subj_acc'].values[0]
        overall_compre_label = 0 if overall_compre<overall_median else 1
        compre = sample_labels['acc'].values[0]
        compre_label = 0 if compre < passage_median else 1
        difficulty = sample_labels['difficulty'].values[0]
        difficulty_label = 0 if difficulty<=1 else 1
        native_label = sample_labels['native'].values[0]

        time, x, y, pupil  = [], [], [], []
        scanpath_tokens, scanpath_token_ids, scanpath_time =[], [], []
        for index,row in sample_df.iterrows():
            fix_x=row['CURRENT_FIX_X']
            fix_y=row['CURRENT_FIX_Y']
            fix_pupil=row['CURRENT_FIX_PUPIL']
            fix_duration=row['CURRENT_FIX_DURATION']

            if index != 0:
                pre_sac_amp=row['PREVIOUS_SAC_AMPLITUDE']
                pre_sac_vel=row['PREVIOUS_SAC_AVG_VELOCITY']
                try:
                    pre_sac_duration=int(pre_sac_amp/pre_sac_vel*1000)
                except Exception as e:
                    pre_sac_duration=0

            past_time=0 if index == 0 else time[-1]+pre_sac_duration+1
            fix_time=[i+past_time for i in range(fix_duration)]
            time+=fix_time
            x+=[fix_x]*fix_duration
            y+=[fix_y]*fix_duration
            pupil+=[fix_pupil]*fix_duration

            fix_iai=row['CURRENT_FIX_INTEREST_AREA_ID']
            if not np.isnan(fix_iai) and fix_iai>3:
                token_id=sample_token_map[sample_token_map['original']==fix_iai]
                token_id=eval(token_id['converted'].values[0])
                scanpath_token_ids+=token_id
                scanpath_tokens+=[row['CURRENT_FIX_INTEREST_AREA_LABEL']]*len(token_id)
                scanpath_time+=[time[-1]]*len(token_id)

        data_list[sample_index]['time'] = time
        data_list[sample_index]['x'] = x
        data_list[sample_index]['y'] = y
        data_list[sample_index]['pupil'] = pupil
        row_scale[sample_index]=row_list
        data_list[sample_index]['scan_tokens']=scanpath_tokens
        data_list[sample_index]['scan_token_ids'] = scanpath_token_ids
        data_list[sample_index]['scan_time'] = scanpath_time
        data_list[sample_index]['label'] = {'overall_compre':overall_compre_label,'compre':compre_label,'difficult':difficulty_label,'native':native_label}
        data_list[sample_index]['sample'] = sample_index
        data_list[sample_index]['corpus'] = text
        data_list[sample_index]['book'] = book_name


    return data_list, row_scale


def draw_image(pid, max_time, times, xs, ys, pupils, label, pupil_scale, row_scale,
                override,image_size, grid_layout,
                linestyle, linewidth,
                color_mapping, idx_mapping,min_duration):
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
    bar_num = row_scale
    strat_y = typesetting['start_y']
    for k in range(bar_num):
        color_id=k%2
        plt.axhspan(ymin=strat_y,ymax=strat_y+typesetting['rowledge'],color=color_mapping[f'split_{color_id}'],alpha=0.5)
        strat_y+=typesetting['rowledge']
    plt.plot(ts_time, y_value, linestyle=linestyle, linewidth=linewidth, color=param_color)
    plt.xlim(0,max_time)
    plt.ylim(0,typesetting['max_y'])
    # plt.xticks([])
    plt.xticks(x_ticks, [''] * len(x_ticks))
    plt.yticks([])

    #pupil
    p_value = np.array(pupils).reshape(-1, 1).astype('float64')
    param_color = color_mapping['pupil']
    plt.subplot(grid_height, grid_width, 3)
    plt.plot(ts_time, p_value, linestyle=linestyle, linewidth=linewidth, color=param_color)
    plt.xlim(0,max_time)
    plt.ylim(pupil_scale)
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
        token_id_map=None,
        linestyle="-", linewidth=1,
        override=False,
        outlier=None,
        interpolation=False,
        image_size=None,
        grid_layout=(3, 1),
        patch_num=16
):
    # load data
    data, row_scales = get_data_label(data_path,subject_id,all_label,token_id_map)

    duration_list, pupil_list= [], []
    for d in data:
        duration_list.append(list(d['time'])[-1])
        pupil_list += list(d['pupil'])
    min_duration = min(duration_list)
    max_time = max(duration_list)
    pupil_list = sorted(list(set(pupil_list)))
    min_pupil = pupil_list[0] if pupil_list[0] != -1 else pupil_list[1]
    pupil_scale = [min_pupil, max(pupil_list)]

    # construct the mapping from param to marker, color, and idx
    num_samples = len(data)
    params = list(data[0].keys())
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
    for idx, p in enumerate(data):
        pid = sample_num+idx
        sample_id = p['sample']
        times = p['time']
        xs = p['x']
        ys = p['y']
        pupils = p['pupil']
        label = p['label']
        subject_id = p['subject']
        book = p['book']
        row_scale = row_scales[idx]

        # split scanpath with patch
        scanpath = p['scan_token_ids']
        if len(scanpath)==0:
            continue
        scanpath_times = p['scan_time']
        text = p['corpus']
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
        draw_image(pid, max_time, times, xs, ys, pupils, label, pupil_scale, row_scale,
                                   override,image_size, grid_layout,
                                   linestyle, linewidth,
                                   color_mapping, idx_mapping,min_duration)


        scanpath_label={'id':pid,
                        'text':text,
                        'scanpath':split_scan,
                        'label':label,
                        'subject_id':subject_id,
                        'sample_id':sample_id,
                        'book':book}
        scanpath_labels.append(scanpath_label)

    sample_num+=num_samples
    return scanpath_labels
if __name__ == "__main__":
    root_path = '../raw_data/fixation'
    all_data = pd.read_csv(os.path.join(root_path,'18sat_fixfinal.csv'))
    all_data = all_data[all_data['type']=='reading']
    sub_path_list = all_data['RECORDING_SESSION_LABEL'].unique()

    label = pd.read_csv(os.path.join(root_path,'18sat_labels.csv'))
    overall_median=label['subj_acc'].median()
    passage_median=label['acc'].median()

    with open('../raw_data/stimuli/text.txt','r',encoding='utf-8') as f:
        texts = f.readlines()
    token_id_map = get_text_token_map(texts)
    scanpath_labels=[]
    sample_num=0
    if not os.path.exists('../processed_data'): os.mkdir('../processed_data')
    for sub_id in tqdm(sub_path_list):
        sub_data = all_data[all_data['RECORDING_SESSION_LABEL']==sub_id].reset_index(drop=True)
        sub_label = label[label['subj']==sub_id].reset_index(drop=True)
        scanpath_label=construct_image(
            data_path=sub_data,
            subject_id=sub_id,
            all_label=sub_label,
            token_id_map=token_id_map,
            linestyle="-", linewidth=1,
            override=True,
            outlier=None,
            interpolation=False,
            image_size=(224,224),
            grid_layout=(3, 1),
            patch_num=16
        )
        scanpath_labels+=scanpath_label
    np.save('../processed_data/duration/Scanpath.npy',scanpath_labels)











