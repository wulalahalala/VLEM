import sys
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold

sys.path.insert(0, '/data3/yiwei.ru/eye_movement_image')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import imp
import argparse
from random import seed
import pandas as pd
import numpy as np
from tqdm import tqdm
import collections.abc

import torch
from transformers import *
from sklearn.metrics import *
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from torchvision.transforms import (
    CenterCrop,
    Compose,
    Resize,
    ToTensor,
)

from models.vision_text_dual_encoder.modeling_vision_text_dual_encoder import VisionTextDualEncoderModelForClassification
from models.vision_text_dual_encoder.configuration_vision_text_dual_encoder import VisionTextDualEncoderForClassificationConfig

from transformers import (
    ViTConfig, 
    BertConfig, 
    ViTFeatureExtractor,
    BertTokenizer
)

from datasets import load_dataset
from datasets import load_metric
from datasets import Dataset, Image

from load_data import get_data_split,split_data

def one_hot(y_):
    y_ = y_.reshape(len(y_))

    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


def fine_tune_hf(
    image_model_path,
    text_model_path,
    freeze_vision_model,
    freeze_text_model,
    output_dir,
    train_dataset,
    val_dataset,
    test_dataset,
    image_size,
    patch_size,
    grid_layout,
    num_classes,
    max_length,
    epochs,
    train_batch_size,
    eval_batch_size,
    save_steps,
    logging_steps,
    learning_rate,
    seed,
    save_total_limit,
    do_train,
    continue_training,
    args
    ):  

    patch_row_num = int(image_size[1]/patch_size)
    patch_column_num = int(image_size[0]/patch_size)
    # loading model and feature extractor
    if do_train and not continue_training:
        if image_size and grid_layout and ('vit' in image_model_path or 'swin' in image_model_path):
            model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
            image_model_path, text_model_path, num_classes=num_classes, max_length=max_length,
            vision_image_size=image_size, vision_grid_layout=grid_layout, patch_row_num=patch_row_num, patch_column_num=patch_column_num,
            cca_weight=args.cca_weight
            )
        else:
            model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
            image_model_path, text_model_path, num_classes=num_classes
            )
        print("load model from", image_model_path)
    else:
        # if not train, load the fine-tuned model saved in output_dir
        if os.path.exists(output_dir):
            dir_list = os.listdir(output_dir) # find the latest checkpoint
            latest_checkpoint_idx = 0
            for d in dir_list:
                if "checkpoint" in d:
                    checkpoint_idx = int(d.split("-")[-1])
                    if checkpoint_idx > latest_checkpoint_idx:
                        latest_checkpoint_idx = checkpoint_idx

            if latest_checkpoint_idx > 0 and os.path.exists(os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")):
                ft_model_path = os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")
                vision_text_config = VisionTextDualEncoderForClassificationConfig.from_pretrained(ft_model_path)
                model = VisionTextDualEncoderModelForClassification.from_pretrained(ft_model_path, config=vision_text_config)
                print("load from the last checkpoint", image_model_path)
            else: # don't have a fine-tuned model
                if image_size and grid_layout:
                    model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
                    image_model_path, text_model_path, num_classes=num_classes, vision_image_size=image_size, vision_grid_layout=grid_layout
                    )
                else:
                    model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
                    image_model_path, text_model_path, num_classes=num_classes
                    )
        else:
            if image_size and grid_layout:
                model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
                image_model_path, text_model_path, num_classes=num_classes, vision_image_size=image_size, vision_grid_layout=grid_layout
                )
            else:
                model = VisionTextDualEncoderModelForClassification.from_vision_text_pretrained(
                image_model_path, text_model_path, num_classes=num_classes
                )

    # whether to freeze models
    # for name, parameters in model.named_parameters():
    #     print(name, ':', parameters.size())
    if freeze_vision_model == "True":
        for name, param in model.vision_model.named_parameters():
            param.requires_grad = False
            print("freezed vision model parameter: ", name)
    if freeze_text_model == "True":
        for name, param in model.text_model.named_parameters():
            param.requires_grad = False
            print("freezed text model parameter: ", name)

    # feature_extractor = AutoFeatureExtractor.from_pretrained(image_model_path)
    tokenizer = AutoTokenizer.from_pretrained(text_model_path)

    # define evaluation metric
    def compute_metrics_binary(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions, labels = eval_pred

        metric = load_metric("metric/accuracy.py")
        accuracy = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)["accuracy"]
        metric = load_metric("metric/precision.py")
        precision = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)["precision"]
        metric = load_metric("metric/recall.py")
        recall = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)["recall"]
        metric = load_metric("metric/f1.py")
        f1 = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)["f1"]

        denoms = np.sum(np.exp(predictions), axis=1).reshape((-1, 1))
        probs = np.exp(predictions) / denoms

        auc = roc_auc_score(labels, probs[:, 1])
        aupr = average_precision_score(labels, probs[:, 1])

        balanced_accuracy = balanced_accuracy_score(y_pred=np.argmax(predictions, axis=1),y_true=labels)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auroc": auc, "auprc": aupr, 'balanced_accuracy':balanced_accuracy}

    def compute_metrics_multiclass(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions, labels = eval_pred

        metric = load_metric("metric/accuracy.py")
        accuracy = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels)["accuracy"]
        metric = load_metric("metric/precision.py")
        precision = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="macro")["precision"]
        metric = load_metric("metric/recall.py")
        recall = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="macro")["recall"]
        metric = load_metric("metric/f1.py")
        f1 = metric.compute(predictions=np.argmax(predictions, axis=1), references=labels, average="macro")["f1"]

        denoms = np.sum(np.exp(predictions), axis=1).reshape((-1, 1))
        probs = np.exp(predictions) / denoms

        auc = roc_auc_score(one_hot(labels), probs)
        aupr = average_precision_score(one_hot(labels), probs)

        #EER
        n_classes = len(np.unique(labels))
        eer_label = np.zeros((len(labels),n_classes))
        for j in range(len(labels)):
            label_idx= int(labels[j])
            eer_label[j,label_idx]= 1

        eer_label = eer_label.reshape(-1)
        score = probs.reshape(-1)
        fpr, tpr , thresholds = roc_curve(eer_label,score)
        fnr = 1 - tpr
        eer_1 = fpr[np.nanargmin(np.absolute((fnr-fpr)))]
        eer_2 = fpr[np.nanargmin(np.absolute((fnr-fpr)))]
        eer = (eer_1+eer_2)/2

        desired_far_arr=[0.1 , 0.001 , 0.0001]
        frr_at_far=[]
        for desired_far in desired_far_arr:
            if fpr[-1] >= desired_far:
                frr_at_far.append(np.mean(interp1d(fpr, fnr)(desired_far)))


        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auroc": auc, "auprc": aupr, 'eer':eer, 'frr@far1':frr_at_far[0],'frr@far2':frr_at_far[1],'frr@far3':frr_at_far[2]}
    
    def compute_metrics_multilabel(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions, labels = eval_pred

        probs = 1 / (1 + np.exp(-predictions))
        for threshold in np.arange(0.1,1,0.1):
            prelabels = (probs>threshold).astype(int)
            t_accuracy = np.mean(prelabels == labels)
            t_precision=precision_score(y_true=labels, y_pred=prelabels, average='samples')
            t_recall=recall_score(y_true=labels, y_pred=prelabels, average='samples')
            t_f1=f1_score(y_true=labels, y_pred=prelabels, average='samples')
            print(f'threshold:{threshold}, accuracy:{t_accuracy}, precision:{t_precision}, recall:{t_recall}, f1:{t_f1}\n')
            if threshold==0.5:
                accuracy=t_accuracy
                precision=t_precision
                recall=t_recall
                f1=t_f1

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def compute_metrics_regression(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions, labels = eval_pred

        rmse = mean_squared_error(labels, predictions, squared=False)
        mape = mean_absolute_percentage_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)

        return {"rmse": rmse, "mape": mape, "mae": mae}

    train_transforms = Compose(
            [   
                # Resize(feature_extractor.size),
                ToTensor(),
                # Cutout(n_holes=cutout_num,length=cutout_size),
            ]
        )
    val_transforms = Compose(
            [
                # Resize(feature_extractor.size),
                ToTensor(),
            ]
        )

    def preprocess_train(example_batch):
        """Apply train_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
            ]
        text=args.dataset_task+": "+example_batch["text"][0] if args.prompt else example_batch["text"][0]
        text_embeddings = tokenizer(
            text,
            padding='max_length', 
            max_length=max_length,
            truncation=True,
            return_tensors="pt")
        word_ids_sn = text_embeddings.word_ids()
        word_ids_sn = [0]+[val+1 if val is not None else np.nan for val in word_ids_sn[1:]]

        gaze_token_pos, patch_gaze_nums = [], []
        total_gaze_num = 0
        if args.prompt:
                prompt_embeddings=tokenizer(args.dataset_task+": ",padding='max_length', max_length=max_length,truncation=True,return_tensors="pt")
                prompt_ids =prompt_embeddings.word_ids()
                insert_token_num = len(list(set(prompt_ids)))
        else:
                insert_token_num =1

        for patch_scan in example_batch['scanpath'][0]:
            # insert [CLS] (prompt) in scanpath
            patch_scan = [i + insert_token_num for i in patch_scan]
            patch_gaze_token_pos = [np.where(np.array(word_ids_sn) == pos)[0].tolist() for pos in patch_scan]
            # flatten the list
            patch_gaze_token_pos = [item for sublist in patch_gaze_token_pos for item in sublist]
            # record gaze num for each patch
            patch_gaze_num = len(patch_gaze_token_pos)
            if patch_gaze_num != 0:
                patch_gaze_nums.append(patch_gaze_num + total_gaze_num-1)
            else:
                if len(patch_gaze_nums)!=0:
                    patch_gaze_nums.append(patch_gaze_nums[-1])
                else:
                    patch_gaze_nums.append(-1)
                # patch_gaze_nums.append(-1)
            total_gaze_num += patch_gaze_num
            gaze_token_pos += patch_gaze_token_pos

        # padding
        # gaze_token_pos.extend([max_length - 1] * (max_length - len(gaze_token_pos)))
        gaze_token_pos.extend([max_length - 1] * (1500 - len(gaze_token_pos)))    
        example_batch["patch_gaze_num"] = [patch_gaze_nums]
        example_batch["gaze_pos"] = [gaze_token_pos]
        example_batch["input_ids"] = text_embeddings["input_ids"]
        example_batch["attention_mask"] = text_embeddings["attention_mask"]
        example_batch["token_type_ids"] = text_embeddings["token_type_ids"]
        return example_batch

    def preprocess_val(example_batch):
        """Apply val_transforms across a batch."""
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB")) for image in example_batch["image"]
            ]
        text=args.dataset_task+": "+example_batch["text"][0] if args.prompt else example_batch["text"][0]
        text_embeddings = tokenizer(
            text,
            padding='max_length', 
            max_length=max_length,
            truncation=True,
            return_tensors="pt")
        word_ids_sn = text_embeddings.word_ids()
        word_ids_sn = [0]+[val+1 if val is not None else np.nan for val in word_ids_sn[1:]]

        gaze_token_pos, patch_gaze_nums = [], []
        total_gaze_num = 0
        if args.prompt:
                prompt_embeddings=tokenizer(args.dataset_task+": ",padding='max_length', max_length=max_length,truncation=True,return_tensors="pt")
                prompt_ids =prompt_embeddings.word_ids()
                insert_token_num = len(list(set(prompt_ids)))
        else:
                insert_token_num =1

        for patch_scan in example_batch['scanpath'][0]:
            # insert [CLS] (prompt) in scanpath
            patch_scan = [i + insert_token_num for i in patch_scan]
            patch_gaze_token_pos = [np.where(np.array(word_ids_sn) == pos)[0].tolist() for pos in patch_scan]
            # flatten the list
            patch_gaze_token_pos = [item for sublist in patch_gaze_token_pos for item in sublist]
            # record gaze num for each patch
            patch_gaze_num = len(patch_gaze_token_pos)
            if patch_gaze_num != 0:
                patch_gaze_nums.append(patch_gaze_num + total_gaze_num-1)
            else:
                if len(patch_gaze_nums)!=0:
                    patch_gaze_nums.append(patch_gaze_nums[-1])
                else:
                    patch_gaze_nums.append(-1)
                # patch_gaze_nums.append(-1)
            total_gaze_num += patch_gaze_num
            gaze_token_pos += patch_gaze_token_pos

        # padding
        # gaze_token_pos.extend([max_length - 1] * (max_length - len(gaze_token_pos)))
        gaze_token_pos.extend([max_length - 1] * (1500 - len(gaze_token_pos))) 
        example_batch["patch_gaze_num"] = [patch_gaze_nums]
        example_batch["gaze_pos"] = [gaze_token_pos]
        example_batch["input_ids"] = text_embeddings["input_ids"]
        example_batch["attention_mask"] = text_embeddings["attention_mask"]
        example_batch["token_type_ids"] = text_embeddings["token_type_ids"]
        return example_batch

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = torch.stack([example["input_ids"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        token_type_ids = torch.stack([example["token_type_ids"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        gaze_pos = torch.tensor([example["gaze_pos"] for example in examples])
        patch_gaze_num = torch.tensor([example["patch_gaze_num"] for example in examples])
        sample_id = torch.tensor([example["sample_id"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids":token_type_ids, "gaze_pos":gaze_pos, "patch_gaze_num":patch_gaze_num, "labels": labels, "sample_id":sample_id}

    # transform the dataset
    train_dataset.set_transform(preprocess_train)
    val_dataset.set_transform(preprocess_val)
    test_dataset.set_transform(preprocess_val)

    if num_classes == 1:
        compute_metrics = compute_metrics_regression
    elif num_classes == 2:
        compute_metrics = compute_metrics_binary
    elif num_classes > 2:
        compute_metrics =compute_metrics_multiclass if num_classes!=11 else compute_metrics_multilabel

    best_metric=args.best_metric
    greater_is_better = False if args.dataset=='gazebase' else True
    # training arguments
    training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    num_train_epochs=epochs,              # total number of training epochs
    per_device_train_batch_size=train_batch_size,  # batch size per device during training
    per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
    evaluation_strategy = "steps",
    save_strategy = "steps",
    learning_rate=learning_rate, # 2e-5
    gradient_accumulation_steps=4,
    warmup_ratio=0.1,
    # weight_decay=0.05,
    # lr_scheduler_type="cosine",
    # fp16=True,
    # fp16_backend="amp",
    save_steps=save_steps,
    logging_steps=logging_steps,
    logging_dir=os.path.join(output_dir, "runs/"),
    save_total_limit=save_total_limit,
    seed=seed,
    load_best_model_at_end=True,
    remove_unused_columns=False,
    dataloader_drop_last=True,
    metric_for_best_model=best_metric, # use loss if not specified
    greater_is_better=greater_is_better
    )

    trainer = Trainer(
    model,
    training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # eval_dataset=test_dataset,
    # tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    # training the model with Huggingface 🤗 trainer
    if do_train:
        train_results = trainer.train()
        # trainer.save_model()
        # trainer.log_metrics("train", train_results.metrics)
        # trainer.save_metrics("train", train_results.metrics)
        # trainer.save_state()
    
    # testing results
    predictions = trainer.predict(test_dataset)
    logits, labels = predictions.predictions, predictions.label_ids
    ypred = np.argmax(logits, axis=1)
    denoms = np.sum(np.exp(logits), axis=1).reshape((-1, 1))
    probs = np.exp(logits) / denoms

    if num_classes == 1:
        acc = precision = recall = F1 = auc = aupr = balanced_accuracy = 0.
        rmse = mean_squared_error(labels, logits, squared=False)
        mape = mean_absolute_percentage_error(labels, logits)
        mae = mean_absolute_error(labels, logits)

    elif num_classes == 2:
        acc = np.sum(labels.ravel() == ypred.ravel()) / labels.shape[0]
        precision = precision_score(labels, ypred)
        recall = recall_score(labels, ypred)
        F1 = f1_score(labels, ypred)
        auc = roc_auc_score(labels, probs[:, 1])
        aupr = average_precision_score(labels, probs[:, 1])
        rmse = mape = mae = eer = 0.
        frr_at_far = [0.,0.,0.]
        balanced_accuracy = balanced_accuracy_score(y_pred=ypred,y_true=labels)

    elif num_classes > 2:
        if num_classes!=11:
            acc = np.sum(labels.ravel() == ypred.ravel()) / labels.shape[0]
            precision = precision_score(labels, ypred, average="macro")
            recall = recall_score(labels, ypred, average="macro") 
            F1 = f1_score(labels, ypred, average="macro")
            auc = roc_auc_score(one_hot(labels), probs)
            aupr = average_precision_score(one_hot(labels), probs)
            rmse = mape = mae = 0.
            balanced_accuracy = balanced_accuracy_score(y_pred=ypred,y_true=labels)
            #eer
            eer_arr = []
            frr_at_far = []
            desired_far_arr = [0.1,0.001,0.0001]

            n_classes = len(np.unique(labels))
            eer_label = np.zeros((len(labels),n_classes))
            for j in range(len(labels)):
                label_idx= int(labels[j])
                eer_label[j,label_idx]= 1

            eer_label = eer_label.reshape(-1)
            score = probs.reshape(-1)
            fpr, tpr , thresholds = roc_curve(eer_label,score)
            fnr = 1 - tpr
            eer_1 = fpr[np.nanargmin(np.absolute((fnr-fpr)))]
            eer_2 = fpr[np.nanargmin(np.absolute((fnr-fpr)))]
            eer = (eer_1+eer_2)/2

            for desired_far in desired_far_arr:
                if fpr[-1] >= desired_far:
                    frr_at_far.append(np.mean(interp1d(fpr, fnr)(desired_far)))

        else:
            probs = 1 / (1 + np.exp(-logits))
            ypred = (probs>0.5).astype(int)
            acc = np.mean(labels == ypred)
            precision = precision_score(labels, ypred, average="samples")
            recall = recall_score(labels, ypred, average="samples") 
            F1 = f1_score(labels, ypred, average="samples")
            auc = aupr = rmse = mape = mae = 0.
            balanced_accuracy = balanced_accuracy_score(y_pred=ypred,y_true=labels)


    return acc, precision, recall, F1, auc, aupr, rmse, mape, mae, balanced_accuracy, eer, frr_at_far


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # arguments for dataset
    parser.add_argument('--dataset', type=str, default='zuco',choices=['zuco','SB-SAT','CopCo'])
    parser.add_argument('--dataset_prefix', type=str, default='duration_3_1_224_224_1') 
    parser.add_argument('--dataset_task', type=str, default='sentiment analysis',
                        choices=['sentiment analysis','general comprehension','comprehension','difficulty','native','dyslexia','sarcasm detection','biometrics','hallucination detection']) #
    parser.add_argument('--split_type', type=str, default='page',choices=['page','book','reader']) # SB-SAT only

    # arguments for huggingface training
    parser.add_argument('--image_model', type=str, default='vit') #
    parser.add_argument('--image_model_path', type=str, default=None)
    parser.add_argument('--text_model', type=str, default='bert_base_uncased')
    parser.add_argument('--text_model_path', type=str, default=None)
    parser.add_argument('--max_length', type=int, default=300, help="Language model's max input length") 
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1799)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--logging_steps', type=int, default=12)
    parser.add_argument('--save_steps', type=int, default=12)

    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--n_splits', type=int, default=5, help='K-fold')
    parser.add_argument('--upsample', default=False)#待办
    parser.add_argument('--binary', default=False, help='turn the CRI dataset into 2 class')

    # argument for the images
    parser.add_argument('--grid_layout', default=None)
    parser.add_argument('--image_size', default=None)
    parser.add_argument('--mask_patch_size', type=int, default=None)
    parser.add_argument('--mask_ratio', type=float, default=None)
    parser.add_argument('--mask_method', type=str, default=None)

    # argument for ablation study
    parser.add_argument('--do_train', default=True)
    parser.add_argument('--finetune_mim', action='store_true')
    parser.add_argument('--freeze_vision_model', type=str, default="False")
    parser.add_argument('--freeze_text_model', type=str, default="False")
    parser.add_argument('--continue_training', action='store_true')

    parser.add_argument('--prompt', default=False)
    parser.add_argument('--cca_weight', type=float, default=0.2)

    args = parser.parse_args()

    dataset = args.dataset
    dataset_prefix = args.dataset_prefix
    print(f'Dataset used: {dataset}, prefix: {dataset_prefix}.')
    
    upsample = args.upsample
    epochs = args.epochs
    grid_layout = None
    mask_patch_size = args.mask_patch_size
    mask_ratio = args.mask_ratio
    mask_method = args.mask_method
    freeze_vision_model = args.freeze_vision_model
    freeze_text_model = args.freeze_text_model
    max_length=args.max_length
    train_batch_size=args.train_batch_size
    data_type=dataset_prefix.split("_")[0]
    if dataset == 'zuco':
        base_path = os.path.join('dataset/zuco',args.dataset_task,data_type)
        num_classes = 3 if args.dataset_task=='sentiment analysis' else 11
        image_size = (int(dataset_prefix.split("_")[3]),int(dataset_prefix.split("_")[4]))
        grid_layout = (3,1)
        task='classification'
        split_key='text'
        n_splits=10
        text_model="bert_base_uncased"
        args.best_metric="f1"
    elif dataset == 'SB-SAT':
        base_path = os.path.join('dataset/SB-SAT',data_type)
        num_classes = 2
        image_size = (int(dataset_prefix.split("_")[3]),int(dataset_prefix.split("_")[4]))
        grid_layout = (3,1)
        task='classification'
        if args.split_type=='page':
            split_key='text'
            n_splits=5
        elif args.split_type=='book':
            split_key='book'
            n_splits=4
        else:
            split_key='subject_id'
            n_splits=5
        text_model="bert_base_uncased"
        args.best_metric="auroc"
    elif dataset == 'CopCo':
        base_path = os.path.join('dataset/CopCo',data_type)
        num_classes = 2
        image_size = (int(dataset_prefix.split("_")[3]),int(dataset_prefix.split("_")[4]))
        grid_layout = (2,1)
        task='classification'
        split_key='undersample'
        n_splits=10
        text_model="bert-base-multilingual-cased"
        args.best_metric="f1"

   



    """prepare the model for vision-text classification"""
    image_model = args.image_model
    if image_model == "vit": # default vit
        image_model_path = "models/vit/vit-base-patch16-224-in21k"
        patch_size = 16
    elif image_model == "vit-384":
        image_model_path = "google/vit-base-patch16-384"
        patch_size = 16
    elif image_model == "swin": # default swin
        image_model_path = "microsoft/swin-base-patch4-window7-224-in22k"
        patch_size = 4
    elif image_model == "swin-224":
        image_model_path = "microsoft/swin-base-patch4-window7-224"
        patch_size = 4

    if text_model == "bert-base-multilingual-cased":
        text_model_path = "models/bert/bert-base-multilingual-cased"
    elif text_model == "bert_base_uncased":
        text_model_path = "models/bert/bert_base_uncased"

    """prepare for training"""
    n_runs = args.n_runs
    acc_arr = np.zeros((n_splits, n_runs))
    auprc_arr = np.zeros((n_splits, n_runs))
    auroc_arr = np.zeros((n_splits, n_runs))
    precision_arr = np.zeros((n_splits, n_runs))
    recall_arr = np.zeros((n_splits, n_runs))
    F1_arr = np.zeros((n_splits, n_runs))
    rmse_arr = np.zeros((n_splits, n_runs))
    mape_arr = np.zeros((n_splits, n_runs))
    mae_arr = np.zeros((n_splits, n_runs))
    balanced_acc_arr = np.zeros((n_splits, n_runs))
    eer_arr = np.zeros((n_splits, n_runs))
    frr_at_far_arr1 = np.zeros((n_splits, n_runs))
    frr_at_far_arr2 = np.zeros((n_splits, n_runs))
    frr_at_far_arr3 = np.zeros((n_splits, n_runs))

    data_list=np.load(os.path.join(base_path,'Scanpath.npy'),allow_pickle=True)
    train_val_test_index=split_data(data_list,n_splits,args.dataset,split_key)
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    fold_index=-1
    for train_idx, val_idx, test_idx in train_val_test_index:
        fold_index+=1
        split_idx = fold_index + 1
        if split_idx<0:
            continue
        print('Split id: %d' % split_idx)

        # fine the pretrained mim image model
        if args.finetune_mim:
            pretrained_image_model_dir = f"../../ckpt/ImgMIM/{dataset_prefix}{dataset}_{image_model}_{mask_patch_size}_{mask_ratio}_{mask_method}/split{split_idx}"
            if os.path.exists(pretrained_image_model_dir):
                dir_list = os.listdir(pretrained_image_model_dir) # find the latest checkpoint
                latest_checkpoint_idx = 0
                for d in dir_list:
                    if "checkpoint" in d:
                        checkpoint_idx = int(d.split("-")[-1])
                        if checkpoint_idx > latest_checkpoint_idx:
                            latest_checkpoint_idx = checkpoint_idx

                if latest_checkpoint_idx > 0 and os.path.exists(os.path.join(pretrained_image_model_dir, f"checkpoint-{latest_checkpoint_idx}")):
                    image_model_path = os.path.join(pretrained_image_model_dir, f"checkpoint-{latest_checkpoint_idx}")

        # the path to save models
        gpu_num=torch.cuda.device_count()
        total_train_batch_size=gpu_num*train_batch_size
        if args.output_dir is None:
            if args.finetune_mim:
                output_dir = f"ckpt/VisionTextPresent/{dataset_prefix}{dataset}_{image_model}_{text_model}_mim_{mask_patch_size}_{mask_ratio}_{mask_method}/split{split_idx}"
            else:
                # output_dir = f"ckpt/{args.dataset}/VisionTextPresent/{args.dataset_task}/PLM_AS_{dataset_prefix}{dataset}_{text_model}_{total_train_batch_size}_GRU_scanpath_maxlength512_repeat1/split{split_idx}"
                if args.dataset=='SB-SAT':
                    output_dir = f"ckpt/{args.dataset}/VisionTextPresent/{args.split_type}/{args.dataset_task}/scratch_{dataset_prefix}{dataset}_{image_model}_{text_model}_{total_train_batch_size}_GRU_scanpath_{args.cca_weight}CCA_WD_15layernormal/split{split_idx}"
                else:
                    output_dir = f"ckpt/debug"
                    # output_dir = f"ckpt/{args.dataset}/VisionTextPresent/{args.dataset_task}/scratch_{dataset_prefix}{dataset}_{image_model}_{text_model}_{total_train_batch_size}_GRU_scanpath_{args.cca_weight}CCA_WD_15layernormal/split{split_idx}"
        else:
            output_dir = f"ckpt/VisionTextPresent/{args.output_dir}/split{split_idx}"

        # prepare the data:
        Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(base_path, data_list, train_idx, test_idx, task, idx_val=val_idx, dataset=args.dataset ,binary=args.binary, data_task=args.dataset_task, prefix=dataset_prefix,upsample=upsample)
        print(len(Ptrain), len(Pval), len(Ptest), len(ytrain), len(yval), len(ytest))

        for m in range(n_runs):
            print('- - Run %d - -' % (m + 1))
            acc, precision, recall, F1, auc, aupr, rmse, mape, mae, balanced_acc, eer, frr_at_far= fine_tune_hf(
            image_model_path=image_model_path,
            text_model_path=text_model_path,
            freeze_vision_model=freeze_vision_model,
            freeze_text_model=freeze_text_model,
            output_dir=output_dir,
            train_dataset=Ptrain,
            val_dataset=Pval,
            test_dataset=Ptest,
            image_size=image_size,
            patch_size=patch_size,
            grid_layout=grid_layout,
            num_classes=num_classes,
            max_length=max_length,
            epochs=epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            learning_rate=args.learning_rate,
            seed=args.seed,
            save_total_limit=args.save_total_limit,
            do_train=args.do_train,
            continue_training=args.continue_training,
            args=args
            )

            test_report = 'Testing: Precision = %.2f | Recall = %.2f | F1 = %.2f\n' % (precision * 100, recall * 100, F1 * 100)
            test_report += 'Testing: AUROC = %.2f | AUPRC = %.2f | Accuracy = %.2f\n' % (auc * 100, aupr * 100, acc * 100)
            test_report += 'Testing: RMSE = %.2f | MAPE = %.2f | MAE = %.2f\n' % (rmse, mape, mae)
            test_report += 'Testing: Balanced Accuracy = %.2f\n' % (balanced_acc * 100)
            test_report += 'Testing: EER = %.4f | FRR@FAR10-1 = %.4f | FRR@FAR10-2 = %.4f | FRR@FAR10-3 = %.4f\n' % (eer ,frr_at_far[0],frr_at_far[1],frr_at_far[2])
            print(test_report)

            if args.do_train:
                result_path = "train_result.txt"
            else:
                result_path = "test_result.txt"
            with open(os.path.join(output_dir, result_path), "w+") as f:
                f.write(test_report)

            # store testing results
            acc_arr[fold_index, m] = acc * 100
            auprc_arr[fold_index, m] = aupr * 100
            auroc_arr[fold_index, m] = auc * 100
            precision_arr[fold_index, m] = precision * 100
            recall_arr[fold_index, m] = recall * 100
            F1_arr[fold_index, m] = F1 * 100
            rmse_arr[fold_index, m] = rmse
            mape_arr[fold_index, m] = mape
            mae_arr[fold_index, m] = mae
            balanced_acc_arr[fold_index, m] = balanced_acc * 100
            eer_arr[fold_index, m] = eer
            frr_at_far_arr1[fold_index, m] = frr_at_far[0]
            frr_at_far_arr2[fold_index, m] = frr_at_far[1]
            frr_at_far_arr3[fold_index, m] = frr_at_far[2]

        
    # pick best performer for each split based on max F1
    idx_max = np.argmax(F1_arr, axis=1)
    acc_vec = [acc_arr[k, idx_max[k]] for k in range(n_splits)]
    auprc_vec = [auprc_arr[k, idx_max[k]] for k in range(n_splits)]
    auroc_vec = [auroc_arr[k, idx_max[k]] for k in range(n_splits)]
    precision_vec = [precision_arr[k, idx_max[k]] for k in range(n_splits)]
    recall_vec = [recall_arr[k, idx_max[k]] for k in range(n_splits)]
    F1_vec = [F1_arr[k, idx_max[k]] for k in range(n_splits)]
    rmse_vec = [rmse_arr[k, idx_max[k]] for k in range(n_splits)]
    mape_vec = [mape_arr[k, idx_max[k]] for k in range(n_splits)]
    mae_vec = [mae_arr[k, idx_max[k]] for k in range(n_splits)]
    balanced_acc_vec = [balanced_acc_arr[k, idx_max[k]] for k in range(n_splits)]
    eer_vec = [eer_arr[k, idx_max[k]] for k in range(n_splits)]
    frr_at_far_vec1 = [frr_at_far_arr1[k, idx_max[k]] for k in range(n_splits)]
    frr_at_far_vec2 = [frr_at_far_arr2[k, idx_max[k]] for k in range(n_splits)]
    frr_at_far_vec3 = [frr_at_far_arr3[k, idx_max[k]] for k in range(n_splits)]

    mean_acc, std_acc = np.mean(acc_vec), np.std(acc_vec)
    mean_auprc, std_auprc = np.mean(auprc_vec), np.std(auprc_vec)
    mean_auroc, std_auroc = np.mean(auroc_vec), np.std(auroc_vec)
    mean_precision, std_precision = np.mean(precision_vec), np.std(precision_vec)
    mean_recall, std_recall = np.mean(recall_vec), np.std(recall_vec)
    mean_F1, std_F1 = np.mean(F1_vec), np.std(F1_vec)
    mean_rmse, std_rmse = np.mean(rmse_vec), np.std(rmse_vec)
    mean_mape, std_mape = np.mean(mape_vec), np.std(mape_vec)
    mean_mae, std_mae = np.mean(mae_vec), np.std(mae_vec)
    mean_balanced_acc, std_balanced_acc = np.mean(balanced_acc_vec), np.std(balanced_acc_vec)
    mean_eer, std_eer = np.mean(eer_vec), np.std(eer_vec)
    mean_frr_at_far1, std_frr_at_far1 = np.mean(frr_at_far_vec1), np.std(frr_at_far_vec1)
    mean_frr_at_far2, std_frr_at_far2 = np.mean(frr_at_far_vec2), np.std(frr_at_far_vec2)
    mean_frr_at_far3, std_frr_at_far3 = np.mean(frr_at_far_vec3), np.std(frr_at_far_vec3)

    # printing the report
    test_report += '------------------------------------------\n'
    test_report += 'Accuracy      = %.2f +/- %.2f\n' % (mean_acc, std_acc)
    test_report += 'AUPRC         = %.2f +/- %.2f\n' % (mean_auprc, std_auprc)
    test_report += 'AUROC         = %.2f +/- %.2f\n' % (mean_auroc, std_auroc)
    test_report += 'Precision     = %.2f +/- %.2f\n' % (mean_precision, std_precision)
    test_report += 'Recall        = %.2f +/- %.2f\n' % (mean_recall, std_recall)
    test_report += 'F1            = %.2f +/- %.2f\n' % (mean_F1, std_F1)
    test_report += 'RMSE          = %.2f +/- %.2f\n' % (mean_rmse, std_rmse)
    test_report += 'MAPE          = %.2f +/- %.2f\n' % (mean_mape, std_mape)
    test_report += 'MAE           = %.2f +/- %.2f\n' % (mean_mae, std_mae)
    test_report += 'Balanced Accuracy = %.2f +/- %.2f\n' % (mean_balanced_acc, std_balanced_acc)
    test_report += 'EER           = %.4f +/- %.4f\n' % (mean_eer, std_eer)
    test_report += 'FRR@FAR10-1   = %.4f +/- %.4f\n' % (mean_frr_at_far1, std_frr_at_far1)
    test_report += 'FRR@FAR10-2   = %.4f +/- %.4f\n' % (mean_frr_at_far2, std_frr_at_far2)
    test_report += 'FRR@FAR10-3   = %.4f +/- %.4f\n' % (mean_frr_at_far3, std_frr_at_far3)
    print(test_report)

    with open(os.path.join(output_dir.split("split")[0], "test_result.txt"), "w+") as f:
        f.write(test_report)


