import ast
import json
import os
import re
import string
import time
from datetime import datetime, timedelta
from os import path

import more_itertools as mit
import numpy as np
import pandas as pd
import torch
import wandb
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from Models.bertModels import SC_weighted_BERT
from src.models.utils import fix_the_random, format_time, masked_cross_entropy, save_bert_model, save_normal_model, get_gpu, load_model
from Preprocess.dataCollect import get_annotated_data, convert_data, get_test_data
import torchaudio

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from src.data.dataLoader import combine_features
from src.data.datsetSplitter import createDatasetSplit, encodeData
from tqdm import tqdm
from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

dict_data_folder = {
    '2': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes_two.npy'},
    '3': {'data_file': 'Data/dataset.json', 'class_label': 'Data/classes.npy'}
}


def select_model(params, embeddings):
    if params["bert_tokens"]:
        if params["what_bert"] == "weighted":
            model = SC_weighted_BERT.from_pretrained(
                # Use the 12-layer BERT model, with an uncased vocab.
                params["path_files"],
                # The number of output labels
                num_labels=params["num_classes"],
                # Whether the model returns attentions weights.
                output_attentions=True,
                # Whether the model returns all hidden-states.
                output_hidden_states=False,
                hidden_dropout_prob=params["dropout_bert"],
                params=params,
            )
        else:
            print("Error in bert model name!!!!")
        return model
    else:
        print("Error in model name!!!!")
        return model


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def Eval_phase(params, which_files="test", model=None, test_dataloader=None, device=None, doc_ids=None, wer_datapoint_graph=False):
    if params["is_model"] == True:
        print("model previously passed")
        model.eval()
    else:
        return 1
    #         ### Have to modify in the final run
    #         model=select_model(params['what_bert'],params['path_files'],params['weights'])
    #         model.cuda()
    #         model.eval()

    print("Running eval on ", which_files, "...")
    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables

    true_labels = []
    pred_labels = []
    logits_all = []
    attention_all = []
    input_mask_all = []

    input_ids_all = []
    # ids_list = list(test_data['Post_id'])
    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()
        outputs = model(b_input_ids, attention_vals=b_att_val, attention_mask=b_input_mask, labels=None, device=device)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()
        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)

        # get rationales from attention
        attention_vectors = np.mean(
            outputs[1][11][:, :, 0, :].detach().cpu().numpy(), axis=1)       
        
        # pred_label == normal -> make attention into zero
        ###################################################################
        pred_label_list = np.argmax(logits, axis=1).flatten()
        for idx, pred_label in enumerate(pred_label_list):
            if pred_label == 0:
                attention_vectors[idx] = [0] * len(attention_vectors[0])
        ###################################################################

        attention_all += list(attention_vectors)
        input_mask_all += list(batch[2].detach().cpu().numpy())

        input_ids_all += list(b_input_ids.detach().cpu().numpy())


    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    attention_vector_final = []
    input_ids_final = []

    for x, y, z in zip(attention_all, input_mask_all, input_ids_all):
        temp = []
        temp_z = []
        for x_ele, y_ele, z_ele in zip(x, y, z):
            if(y_ele == 1):
                temp.append(x_ele)
                temp_z.append(z_ele)
        attention_vector_final.append(temp)
        input_ids_final.append(temp_z)
    
    pred_soft_attention = {}
    if doc_ids:
        for idx, id in enumerate(doc_ids):
            pred_soft_attention[id] = {'pred_soft_attention': ([float(att) for att in attention_vector_final[idx]]),
                                        'input_ids': [float(input_id) for input_id in input_ids_final[idx]],
                                        }
        pred_hard_attention = get_pred_hard_token_rationales(pred_soft_attention, params)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
        pred_hard_word_rationales = get_pred_hard_word_rationales(pred_hard_attention, tokenizer, params)
        pred_timestamp_dict = get_pred_timestamp(pred_hard_word_rationales, params)

        # audio input
        # with open('/home/jinmyeong/code/hts/classifier/Data/audio_rationales_data.json', 'r') as fp:
        #     audioHateXplain = json.load(fp)
        with open(params["gold_transcript_data_path"], 'r') as fp:
            audioHateXplain = json.load(fp)

        filterd_gold_timestamp_list = get_filtered_gold_timestamp_list(audioHateXplain, pred_timestamp_dict)
        last_time_list = get_last_time_list(pred_timestamp_dict, params)
        pred_timestamp_list = [pred_timestamp_dict[key]['pred_timestamp'] for key in pred_timestamp_dict]
        gold_timestamp_binary_list, pred_timestamp_binary_list = get_timestamp_binary_list(last_time_list, filterd_gold_timestamp_list, pred_timestamp_list)
    
        # Save the timestamp result for gold and predicted with id
        id_list = list(pred_timestamp_dict.keys())

        def save_timestamp_binary_list(id_list, gold_timestamp_binary_list, pred_timestamp_binary_list):
            timestamp_result_dict_list = []
            for id, gold_timestamp_binary, pred_timestamp_binary in zip(id_list, gold_timestamp_binary_list, pred_timestamp_binary_list):
                timestamp_result_dict = {}
                timestamp_result_dict['id'] = id
                timestamp_result_dict['gold'] = gold_timestamp_binary
                timestamp_result_dict['pred'] = pred_timestamp_binary
                timestamp_result_dict_list.append(timestamp_result_dict)
            
            with open(f"/home/jinmyeong/code/hts/classifier/Result/{params['data_type']}.jsonl", encoding="utf-8", mode="w") as f:
                for e in timestamp_result_dict_list:
                    f.write(json.dumps(e) + "\n")
            
        save_timestamp_binary_list(id_list, gold_timestamp_binary_list, pred_timestamp_binary_list)


        # calculate IOU
        iou_all_data_all_label = get_IOU(
            pred_timestamp_binary_list, gold_timestamp_binary_list, data_mode="all_data", IOU_mode="all_label")
        if wer_datapoint_graph:
            iou_all_data_hate_label, id_iou_pair = get_IOU(
                pred_timestamp_binary_list, gold_timestamp_binary_list, data_mode="all_data", 
                IOU_mode="hate_label", id_list=id_list, true_labels=true_labels, pred_labels=pred_labels) # IOU for explainability
        iou_all_data_hate_label = get_IOU(
            pred_timestamp_binary_list, gold_timestamp_binary_list, data_mode="all_data", IOU_mode="hate_label") # IOU for explainability

        iou_hate_data_all_label = get_IOU(
            pred_timestamp_binary_list, gold_timestamp_binary_list, data_mode="hate_data", IOU_mode="all_label")

        iou_hate_data_hate_label = get_IOU(
            pred_timestamp_binary_list, gold_timestamp_binary_list, data_mode="hate_data", IOU_mode="hate_label")
        
        # save wer_datapoint_graph
        if wer_datapoint_graph:
            file_path = f"/home/jinmyeong/code/hts/classifier/Data/id_iou_pair_{params['data_type']}.json"
            with open(file_path, "w") as f:
                json.dump(id_iou_pair, f)  
            
            id_label_pair = []
            for idx, id in enumerate(doc_ids):
                true_label, pred_label = int(true_labels[idx]), int(pred_labels[idx])
                id_label = {"id": id, "true_label": true_label, "pred_label": pred_label}
                id_label_pair.append(id_label)
            
            file_path = f"/home/jinmyeong/code/hts/classifier/Data/id_label_pair_{params['data_type']}.json"
            with open(file_path, "w") as f:
                json.dump(id_label_pair, f)  

            # get example for bad cases with cascaded, but good cased with E2E
            # ids under the average IOU
            import jiwer
            ids_0_5 = [id_iou for id_iou in id_iou_pair if id_iou['iou'] < 0.05 and id_iou['iou'] > 0]
            ids_5_10 = [id_iou for id_iou in id_iou_pair if id_iou['iou'] < 0.1 and id_iou['iou'] > 0.05]
            ids_10_15 = [id_iou for id_iou in id_iou_pair if id_iou['iou'] < 0.15 and id_iou['iou'] > 0.1]
            ids_15_20 = [id_iou for id_iou in id_iou_pair if id_iou['iou'] < 0.2 and id_iou['iou'] > 0.15]
            
            def get_gold_final_rationales(key):

                pred_timestamp = pred_timestamp_dict[key]['pred_timestamp']
                gold_timestamp = [[interval["xmin"], interval["xmax"]] for interval in audioHateXplain[key]["intervals"].values()]
                
                rationale_list = audioHateXplain[key]['rationale']

                # if len(gold_timestamp) != len(rationale_list[0]):
                #     print(key)

                if audioHateXplain[key]["intervals"]["intervals [1]"].get('rationale'):
                    final_rationale = [audioHateXplain[key]["intervals"][f"intervals [{idx+1}]"]['rationale'] for idx, _ in enumerate(list(audioHateXplain[key]['intervals'].items()))]
                    # final_rationales.append(final_rationale)
                else:
                    
                    the_len = len(rationale_list[0])
                    final_rationale = []
                    for i in range(0, the_len):
                        summed = 0
                        
                        for j in range(0, len(rationale_list)):
                            summed += int(rationale_list[j][i])
                        
                        if summed/len(rationale_list) >= 0.5:
                            final_rationale.append(1)
                        else:
                            final_rationale.append(0)
                        # final_rationale.append(summed/len(rationale_list))

                return final_rationale
            def get_WER(reference, hypothesis):
                error = jiwer.wer(reference, hypothesis)
                
                output = jiwer.process_words(reference, hypothesis)
                error = output.wer
                return error
            def get_wer(gold_text, transcript):
                error = jiwer.process_words(gold_text, transcript)
                alignments = error.alignments[0]

                gold_text_lower = gold_text.lower().replace('.', '').replace(',', '').replace('<', '').replace('>', '').replace('?', '').replace('-', ' ')
                transcript_lower = transcript.lower().replace('.', '').replace(',', '').replace('<', '').replace('>', '').replace('?', '').replace('-', ' ')

                wer = get_WER(gold_text_lower, transcript_lower)  
                return wer
                          
            import random
            ids_0_5_sample = random.sample(ids_0_5, 5)
            ids_5_10_sample = random.sample(ids_5_10, 5)
            ids_10_15_sample = random.sample(ids_10_15, 5)
            ids_15_20_sample = random.sample(ids_15_20, 5)

            ids_sample = []
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
            ids_sample = ids_sample + ids_0_5_sample + ids_5_10_sample + ids_10_15_sample + ids_15_20_sample
            for id_sample in ids_sample:
                id = id_sample['id']

                id_order = 0
                for idx, id_datapoint in enumerate(id_list):
                    if id == id_datapoint:
                        id_order = idx

                GT_time_frame_rationales = gold_timestamp_binary_list[id_order]
                pred_time_frame_rationales = pred_timestamp_binary_list[id_order]
                
                GT_time_intervals = [] # filterd_gold_timestamp_list[id_order]
                for interval_key in audioHateXplain[id]['intervals'].keys():
                    xmin = audioHateXplain[id]['intervals'][interval_key]['xmin']
                    xmax = audioHateXplain[id]['intervals'][interval_key]['xmax']
                    GT_time_intervals.append([xmin, xmax])
                    
                pred_time_intervals = pred_timestamp_dict[id]['pred_timestamp']

                GT_word_ratinales = get_gold_final_rationales(key=id)
                pred_word_ratinales = pred_timestamp_dict[id]['hard_word_attention']

                GT_word_list = audioHateXplain[id]['sentence']
                pred_word_list = tokenizer.decode(pred_timestamp_dict[id]['input_ids']).split(" ")[1:-1]

                wer = get_wer(gold_text=" ".join(GT_word_list), transcript=" ".join(pred_word_list))

                id_sample['GT_time_frame_rationales'] = GT_time_frame_rationales
                id_sample['pred_time_frame_rationales'] = pred_time_frame_rationales
                
                id_sample['GT_time_intervals'] = GT_time_intervals
                id_sample['pred_time_intervals'] = pred_time_intervals

                id_sample['GT_word_ratinales'] = GT_word_ratinales
                id_sample['pred_word_ratinales'] = pred_word_ratinales

                id_sample['GT_word_list'] = GT_word_list
                id_sample['pred_word_list'] = pred_word_list

                id_sample['wer'] = wer

            ids_sample_file_path = f"/home/jinmyeong/code/hts/classifier/Data/example_for_cascaded_vs_E2E.json"
            with open(ids_sample_file_path, "w") as f:
                json.dump(ids_sample, f)  
            

        #calculate Token F1
        precisions_all_data, recalls_all_data, f1s_all_data = get_token_f1(
            pred_timestamp_binary_list, gold_timestamp_binary_list, data_mode="all_data")
        
        precisions_hate_data, recalls_hate_data, f1s_hate_data = get_token_f1(
            pred_timestamp_binary_list, gold_timestamp_binary_list, data_mode="hate_data")

    
    testf1 = f1_score(true_labels, pred_labels, average="macro")
    testacc = accuracy_score(true_labels, pred_labels)
    if params["num_classes"] == 3:
        testrocauc = roc_auc_score(true_labels, logits_all_final, multi_class="ovo", average="macro")
    else:
        # testrocauc=roc_auc_score(true_labels, logits_all_final,multi_class='ovo',average='macro')
        testrocauc = 0
    testprecision = precision_score(true_labels, pred_labels, average="macro")
    testrecall = recall_score(true_labels, pred_labels, average="macro")

    if params["logging"] != "neptune" or params["is_model"] == True:
        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.2f}".format(testacc))
        print(" Fscore: {0:.2f}".format(testf1))
        print(" Precision: {0:.2f}".format(testprecision))
        print(" Recall: {0:.2f}".format(testrecall))
        print(" Roc Auc: {0:.2f}".format(testrocauc))
        print(" Test took: {:}".format(format_time(time.time() - t0)))
        # print(ConfusionMatrix(true_labels,pred_labels))
    else:
        bert_model = params["path_files"]
        language = params["language"]
        name_one = bert_model + "_" + language

    if doc_ids:
        return (testf1, testacc, testprecision, testrecall, testrocauc, logits_all_final, 
                iou_all_data_all_label, iou_all_data_hate_label, iou_hate_data_all_label, 
                iou_hate_data_hate_label, precisions_all_data, recalls_all_data, f1s_all_data,
                precisions_hate_data, recalls_hate_data, f1s_hate_data)
    
    return testf1, testacc, testprecision, testrecall, testrocauc, logits_all_final
def get_pred_hard_token_rationales(pred_soft_attention, params):
    for key in pred_soft_attention:
        soft_attention = pred_soft_attention[key]['pred_soft_attention']
        if params['att_threshold'] == None or params['att_threshold'] == 'mean':
            threshold = np.mean(soft_attention) # TODO: threshold 변경
        elif params['att_threshold'] == '-4sigma':
            threshold = np.mean(soft_attention) - 4 * np.std(soft_attention)
        elif params['att_threshold'] == '-3sigma':
            threshold = np.mean(soft_attention) - 3 * np.std(soft_attention)
        elif params['att_threshold'] == '-2sigma':
            threshold = np.mean(soft_attention) - 2 * np.std(soft_attention)
        elif params['att_threshold'] == '-sigma':
            threshold = np.mean(soft_attention) - np.std(soft_attention)
        elif params['att_threshold'] == '+sigma':
            threshold = np.mean(soft_attention) + np.std(soft_attention)
        elif params['att_threshold'] == '+2sigma':
            threshold = np.mean(soft_attention) + 2 * np.std(soft_attention)

        pred_soft_attention[key]['pred_hard_attention'] = [1 if att >= threshold else 0 for att in soft_attention ]
        if threshold == 0:
            pred_soft_attention[key]['pred_hard_attention'] = [0 for att in soft_attention ]
        # TODO: majority filter

    return pred_soft_attention

def get_filtered_gold_timestamp_list(audioHateXplain, pred_timestamp_dict):
    pred_timestamp_list = []
    gold_timestamp_list = []

    final_rationales = []

    for key in pred_timestamp_dict:
        pred_timestamp = pred_timestamp_dict[key]['pred_timestamp']
        gold_timestamp = [[interval["xmin"], interval["xmax"]] for interval in audioHateXplain[key]["intervals"].values()]
        
        rationale_list = audioHateXplain[key]['rationale']

        # if len(gold_timestamp) != len(rationale_list[0]):
        #     print(key)

        if audioHateXplain[key]["intervals"]["intervals [1]"].get('rationale'):
            final_rationale = [audioHateXplain[key]["intervals"][f"intervals [{idx+1}]"]['rationale'] for idx, _ in enumerate(list(audioHateXplain[key]['intervals'].items()))]
            final_rationales.append(final_rationale)
        else:
            try:
                the_len = len(rationale_list[0])
                final_rationale = []
                for i in range(0, the_len):
                    summed = 0
                    
                    for j in range(0, len(rationale_list)):
                        summed += int(rationale_list[j][i])
                    
                    final_rationale.append(summed/len(rationale_list))
                final_rationales.append(final_rationale)
            except:
                final_rationales.append(len(gold_timestamp) * [0])

        pred_timestamp_list.append(pred_timestamp)
        gold_timestamp_list.append(gold_timestamp)

    filterd_gold_timestamp_list = []

    for idx, gold_timestamp in enumerate(gold_timestamp_list):
        doc_id = list(pred_timestamp_dict.items())[idx][0]
        if audioHateXplain[doc_id]['label'] == 'normal':
            filterd_gold_timestamp = []
            filterd_gold_timestamp_list.append(filterd_gold_timestamp)
            continue

        filterd_gold_timestamp = []
        for idx_time, timestamp in enumerate(gold_timestamp):
            if final_rationales[idx][idx_time] >= 0.5:
                filterd_gold_timestamp.append(timestamp)
        
        filterd_gold_timestamp_list.append(filterd_gold_timestamp)
    return filterd_gold_timestamp_list

def _make_timegrid_10ms(sound_duration: float):
    # round at 2 decimal places
    sound_duration = np.round(sound_duration, 2)
    total_len = int(sound_duration / 0.01)
    start_timegrid = np.linspace(0, sound_duration, total_len + 1)

    # print(total_len)
    # print(sound_duration)
    # print(start_timegrid.shape)
    dt = start_timegrid[1] - start_timegrid[0]
    end_timegrid = start_timegrid + dt
    return start_timegrid[:total_len]

def get_last_time_list(pred_timestamp, params):
    # audio_filename_list = os.listdir("/mnt/hdd4/jinmyeong/hts_data/LibriTTS")
    # filepath = "/mnt/hdd4/jinmyeong/hts_data/LibriTTS"
    audio_filename_list = os.listdir(params["audio_path"])
    filepath = params["audio_path"]

    audio_filepath_list = [os.path.join(filepath, filename) for filename in audio_filename_list]
    last_time_list = []

    for idx, key in enumerate(pred_timestamp):
        for idx_audio, audio_filename in enumerate(audio_filename_list):
            if audio_filename[:-4] == key:
                info = torchaudio.info(audio_filepath_list[idx_audio])
                last_time = info.num_frames / info.sample_rate

                last_time_list.append(last_time)
    return last_time_list

def get_timestamp_binary_list(last_time_list, filterd_gold_timestamp_list, pred_timestamp_list):
    gold_timestamp_binary_list = []
    pred_timestamp_binary_list = []

    for time_idx, last_time in enumerate(last_time_list):
        gold_timestamp_binary = []
        pred_timestamp_binary = []

        sample_list = _make_timegrid_10ms(last_time)
        for time_frame in sample_list:
            gold_binary = 0
            pred_binary = 0

            for idx in range(len(filterd_gold_timestamp_list[time_idx])):
                if len(filterd_gold_timestamp_list[time_idx][idx]) < 2:
                    gold_start, gold_end = 0, 0
                else:
                    gold_start, gold_end = filterd_gold_timestamp_list[time_idx][idx][0], filterd_gold_timestamp_list[time_idx][idx][1]
                
                if time_frame >= gold_start and time_frame <= gold_end:
                    gold_binary = 1

            for idx in range(len(pred_timestamp_list[time_idx])):
                if len(pred_timestamp_list[time_idx][idx]) < 2:
                    pred_start, pred_end = 0, 0
                else:
                    pred_start, pred_end = pred_timestamp_list[time_idx][idx][0], pred_timestamp_list[time_idx][idx][1]

                if time_frame >= pred_start and time_frame <= pred_end:
                    pred_binary = 1  

            gold_timestamp_binary.append(gold_binary)
            pred_timestamp_binary.append(pred_binary)
        
        gold_timestamp_binary_list.append(gold_timestamp_binary)
        pred_timestamp_binary_list.append(pred_timestamp_binary)
    
    return gold_timestamp_binary_list, pred_timestamp_binary_list

def calculate_ordered_iou(pred, golden, IOU_mode):
    if len(pred) != len(golden):
        raise ValueError("Lists must be of the same length")

    if IOU_mode == "all_label":
        intersection = sum(a == b for a, b in zip(pred, golden))
        union = len(pred)# + len(golden) - intersection
    elif IOU_mode == "hate_label":
        intersection = sum(a == b for a, b in zip(pred, golden) if a == 1)
        union = len([e for e in pred if e==1]) + len([e for e in golden if e==1]) - intersection

    # Calculate IoU
    iou = intersection / union if union != 0 else None
    return iou, intersection

def calculate_token_f1(pred, golden, data_mode="all_data"):
    if len(pred) != len(golden):
        raise ValueError("Lists must be of the same length")
    
    if data_mode == "hate_data":
        if sum(golden) == 0: # golden == normal
            precision, recall, f1 = None, None, None
            return precision, recall, f1
        
    intersection = sum(a == b for a, b in zip(pred, golden) if a == 1)

    precision = intersection / sum(pred) if sum(pred) != 0 else None
    recall = intersection / sum(golden) if sum(golden) != 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision != None and recall != None and (precision + recall) != 0 else None
    return precision, recall, f1

def get_IOU(pred_timestamp_binary_list, gold_timestamp_binary_list, data_mode="all_data", IOU_mode="all_label", id_list=None, true_labels=None, pred_labels=None):
    if id_list != None:
        id_iou_pair = []
    ious = []
    intersections = []

    for idx, (pred_timestamp_binary, gold_timestamp_binary) in enumerate(zip(pred_timestamp_binary_list, gold_timestamp_binary_list)):  
        if data_mode == "hate_data":
            if sum(gold_timestamp_binary) == 0:
                continue

        iou, intersection = calculate_ordered_iou(pred_timestamp_binary, gold_timestamp_binary, IOU_mode)
        if iou != None:
            ious.append(iou)
            if id_list != None:
                id_iou_pair.append({"id": id_list[idx], "iou": iou}) # , "true_label": int(true_labels[idx]), "pred_label": int(pred_labels[idx])
        intersections.append(intersection / len(pred_timestamp_binary))

    print(f"average intersection ratio: {sum(intersections)/len(intersections)}")
    average_ious = sum(ious)/len(ious)

    if id_list != None:
        return average_ious, id_iou_pair
    return average_ious

def get_token_f1(pred_timestamp_binary_list, gold_timestamp_binary_list, data_mode="all_data"):
    precisions = []
    recalls = []
    f1s = []

    for pred_timestamp_binary, gold_timestamp_binary in zip(pred_timestamp_binary_list, gold_timestamp_binary_list):  
        precision, recall, f1 = calculate_token_f1(pred_timestamp_binary, gold_timestamp_binary, data_mode)
        if precision != None:
            precisions.append(precision)
        if recall != None:
            recalls.append(recall)
        if f1 != None:
            f1s.append(f1)

    average_precisions = sum(precisions)/len(precisions)
    average_recalls = sum(recalls)/len(recalls)
    average_f1s = sum(f1s)/len(f1s)

    return average_precisions, average_recalls, average_f1s

# def get_token_f1(pred_timestamp_binary_list, gold_timestamp_binary_list):
#     fp = 0
#     fn = 0
#     tp = 0
#     tn = 0

#     for pred_timestamp_binary, gold_timestamp_binary in zip(pred_timestamp_binary_list, gold_timestamp_binary_list):  
#         for pred_timestamp, gold_timestamp in zip(pred_timestamp_binary, gold_timestamp_binary):
#             if pred_timestamp != gold_timestamp:
#                 if pred_timestamp == 1:
#                     fp += 1
#                 else:
#                     fn += 1
#             else:
#                 if pred_timestamp == 1:
#                     tp += 1
#                 else:
#                     tn += 1      

#     precision = tp / (tp + fp)
#     recall = tp / (tp + fn)

#     f1 = 2 * precision * recall / (precision + recall)

#     result = f"precision: {precision} \n recall: {recall} \n f1: {f1}"
    
#     return precision, recall, f1

def get_pred_hard_word_rationales(pred_hard_attention, tokenizer, params):
    file_path = params["data_file"] # '/home/jinmyeong/code/hts/classifier/Data/new_dataset.json'
    with open(file_path, "r") as f:
        dataset = json.load(f)    
    hard_word_rationales = []

    for key in pred_hard_attention:
        hard_word_rationale = []
        hard_attention = pred_hard_attention[key]
        hard_attention_list = hard_attention['pred_hard_attention'][1:-1]
        input_ids = hard_attention['input_ids'][1:-1]

        word_list = dataset[key]["post_tokens"]
        tokenized_word_list = []
        for word in word_list:
            tokenized_word = tokenizer.encode(word)[1:-1]
            tokenized_word_list.append(tokenized_word)
        len_tokenized_word_list = [len(tokenized_word) for tokenized_word in tokenized_word_list]
        # TODO: 현재는 임시방편
        for idx, tokenized_word in enumerate(tokenized_word_list):
            start_idx, end_idx = sum(len_tokenized_word_list[:idx]), sum(len_tokenized_word_list[:idx+1])
            try:
                ratio = sum(hard_attention_list[start_idx:end_idx]) / len(hard_attention_list[start_idx:end_idx]) 
            except:
                ratio = 0
            if ratio >= 0.5: 
                hard_word_rationale.append(1)
            else: 
                hard_word_rationale.append(0)

        pred_hard_attention[key]['hard_word_attention'] = hard_word_rationale
    
    return pred_hard_attention

def get_pred_timestamp(pred_hard_word_rationales, params):
    # with open('/home/jinmyeong/code/hts/ASR/Data/ASR/whisperX_ASR_transcription_v4-HateSpeech-all-multispeaker.json', 'r') as fp:
    #     transcript_dict = json.load(fp)

    # segment audio ASR 필요
    with open(params["ASR_data_path"], 'r') as fp:
        transcript_dict = json.load(fp)

    for key in pred_hard_word_rationales:
        hard_word_attention = pred_hard_word_rationales[key]['hard_word_attention']
        timestamp = []
        for idx, hard_attention in enumerate(hard_word_attention):
            if hard_attention == 1:
                try:
                    start, end = transcript_dict[key][0]['words'][idx]['start'], transcript_dict[key][0]['words'][idx]['end']
                except:
                    start, end = 0, 0

                timestamp.append((start, end))
        pred_hard_word_rationales[key]['pred_timestamp'] = timestamp

    return pred_hard_word_rationales

def train_model(
        params,
        device,
        train,
        val,
        test,
        model,
        tokenizer):
    kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
    name = kor_time + f"{params['model_name']}+ep={params['epochs']}+bs={params['batch_size']}"

    embeddings = None
    
    if params["auto_weights"]:
        y_test = [ele[2] for ele in test]
        #         print(y_test)
        encoder = LabelEncoder()
        encoder.classes_ = np.load(params["class_names"], allow_pickle=True)
        params["weights"] = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y_test), y=y_test
        ).astype("float32")

    print(params["weights"])
    train_dataloader = combine_features(train, params, is_train=True)
    validation_dataloader = combine_features(val, params, is_train=False)
    test_dataloader = combine_features(test, params, is_train=False)

    val_id_list = [ele[3] for ele in val] 
    test_id_list = [ele[3] for ele in test] 

    # model = select_model(params, embeddings)

    if params["device"] == "cuda":
        model.cuda()
    optimizer = AdamW(
        model.parameters(),
        # args.learning_rate - default is 5e-5, our notebook had 2e-5
        lr=params["learning_rate"],
        # args.adam_epsilon  - default is 1e-8.
        eps=params["epsilon"],
    )

    # Number of training epochs (authors recommend between 2 and 4)
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * params["epochs"]

    # Create the learning rate scheduler.
    if params["bert_tokens"]:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps / 10), num_training_steps=total_steps
        )

    # Set the seed value all over the place to make this reproducible.
    fix_the_random(seed_val=params["random_seed"])
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    if params["bert_tokens"]:
        bert_model = params["path_files"]
        name_one = bert_model
    else:
        name_one = params["model_name"]

    best_val_fscore = 0
    best_test_fscore = 0

    best_val_roc_auc = 0
    best_test_roc_auc = 0

    best_val_precision = 0
    best_test_precision = 0

    best_val_recall = 0
    best_test_recall = 0

    best_val_iou_all_data_hate_label = 0
    
    for epoch_i in range(0, params["epochs"]):
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, params["epochs"]))
        print("Training...")

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in tqdm(enumerate(train_dataloader)):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention vals
            #   [2]: attention mask
            #   [3]: labels
            b_input_ids = batch[0].to(device)
            b_att_val = batch[1].to(device)
            b_input_mask = batch[2].to(device)
            b_labels = batch[3].to(device)

            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()
            outputs = model(
                b_input_ids, attention_vals=b_att_val, attention_mask=b_input_mask, labels=b_labels, device=device
            )

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.

            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            if params["bert_tokens"]:
                scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        print("avg_train_loss", avg_train_loss)

        # fix soft att threshold
        params['att_threshold'] = 'mean'

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        train_fscore, train_accuracy, train_precision, train_recall, train_roc_auc, _ = Eval_phase(
            params, "train", model, train_dataloader, device
        )
        (val_fscore, val_accuracy, val_precision, val_recall, val_roc_auc, val_logits_all_final,
        val_iou_all_data_all_label, val_iou_all_data_hate_label, val_iou_hate_data_all_label, 
        val_iou_hate_data_hate_label, val_precisions_all_data, val_recalls_all_data, val_f1s_all_data,
        val_precisions_hate_data, val_recalls_hate_data, val_f1s_hate_data) = Eval_phase(params, "val", model, validation_dataloader, device, val_id_list)


        (test_fscore, test_accuracy, test_precision, test_recall, test_roc_auc, test_logits_all_final,
        test_iou_all_data_all_label, test_iou_all_data_hate_label, test_iou_hate_data_all_label, 
        test_iou_hate_data_hate_label, test_precisions_all_data, test_recalls_all_data, test_f1s_all_data,
        test_precisions_hate_data, test_recalls_hate_data, test_f1s_hate_data) = Eval_phase(
            params, "test", model, test_dataloader, device, test_id_list)


        if val_fscore > best_val_fscore:
            print(val_fscore, best_val_fscore)
            best_val_fscore = val_fscore
            best_test_fscore = test_fscore
            best_val_roc_auc = val_roc_auc
            best_test_roc_auc = test_roc_auc

            best_val_precision = val_precision
            best_test_precision = test_precision
            best_val_recall = val_recall
            best_test_recall = test_recall

            if params["bert_tokens"]:
                print("Loading BERT tokenizer...")
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
                save_bert_model(model, tokenizer, params)

        # # BEST IOU model
        # if val_iou_all_data_hate_label > best_val_iou_all_data_hate_label:
        #     print(val_iou_all_data_hate_label, best_val_iou_all_data_hate_label)
        #     best_val_iou_all_data_hate_label = val_iou_all_data_hate_label
        #     best_test_fscore = test_fscore
        #     best_val_roc_auc = val_roc_auc
        #     best_test_roc_auc = test_roc_auc

        #     best_val_precision = val_precision
        #     best_test_precision = test_precision
        #     best_val_recall = val_recall
        #     best_test_recall = test_recall

            # if params["bert_tokens"]:
            #     print("Loading BERT tokenizer...")
            #     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=False)
            #     save_bert_model(model, tokenizer, params)

        wandb.log(
            {
                "train_loss": avg_train_loss,
                "train_epoch": epoch_i,
                "val_accuracy": val_accuracy,
                "test_accuracy": test_accuracy,
                "val_fscore": val_fscore,
                "test_fscore": test_fscore,
                "val_roc_auc": val_roc_auc,
                "test_roc_auc": test_roc_auc,
                "val_frame_accuracy": val_iou_all_data_all_label,
                "val_IOU": val_iou_all_data_hate_label,   
                "val_f1s_all_data": val_f1s_all_data,
                "test_frame_accuracy": test_iou_all_data_all_label,
                "test_IOU": test_iou_all_data_hate_label,   
                "test_f1s_all_data": test_f1s_all_data,
            }
        )

    print("best_val_fscore", best_val_fscore)
    print("best_test_fscore", best_test_fscore)
    print("best_val_rocauc", best_val_roc_auc)
    print("best_test_rocauc", best_test_roc_auc)
    print("best_val_precision", best_val_precision)
    print("best_test_precision", best_test_precision)
    print("best_val_recall", best_val_recall)
    print("best_test_recall", best_test_recall)

    print(f"test_accuracy: {test_accuracy}")
    print(f"test_fscore: {test_fscore}" )
    print(f"test_recall: {test_recall}")
    print(f"test_precision: {test_precision}")

    print(f"val_iou_hate_data_all_label: {val_iou_hate_data_all_label}")
    print(f"val_iou_all_data_all_label: {val_iou_all_data_all_label}")
    print(f"val_iou_hate_data_hate_label: {val_iou_hate_data_hate_label}")
    print(f"val_iou_all_data_hate_label: {val_iou_all_data_hate_label}")
    print(f"val_precisions_hate_data: {val_precisions_hate_data}")
    print(f"val_precisions_all_data: {val_precisions_all_data}")
    print(f"val_recalls_hate_data: {val_recalls_hate_data}")
    print(f"val_f1s_hate_data: {val_f1s_hate_data}")

    print(f"test_iou_hate_data_all_label: {test_iou_hate_data_all_label}")
    print(f"test_iou_all_data_all_label: {test_iou_all_data_all_label}")
    print(f"test_iou_hate_data_hate_label: {test_iou_hate_data_hate_label}")
    print(f"test_iou_all_data_hate_label: {test_iou_all_data_hate_label}")
    print(f"test_precisions_hate_data: {test_precisions_hate_data}")
    print(f"test_precisions_all_data: {test_precisions_all_data}")
    print(f"test_recalls_hate_data: {test_recalls_hate_data}")
    print(f"test_f1s_hate_data: {test_f1s_hate_data}")
    del model
    torch.cuda.empty_cache()
    return 1


def get_annotated_data(params):
    # temp_read = pd.read_pickle(params['data_file'])
    with open(params["data_file"], "r") as fp:
        data = json.load(fp)
    dict_data = []
    for key in data:
        temp = {}
        temp["post_id"] = key
        temp["text"] = data[key]["post_tokens"]
        final_label = []
        for i in range(1, 4):
            temp["annotatorid" + str(i)] = data[key]["annotators"][i - 1]["annotator_id"]
            #             temp['explain'+str(i)]=data[key]['annotators'][i-1]['rationales']
            temp["target" + str(i)] = data[key]["annotators"][i - 1]["target"]
            temp["label" + str(i)] = data[key]["annotators"][i - 1]["label"]
            final_label.append(temp["label" + str(i)])

        final_label_id = max(final_label, key=final_label.count)
        temp["rationales"] = data[key]["rationales"]

        if params["class_names"] == "Data/classes_two.npy":
            if final_label.count(final_label_id) == 1:
                temp["final_label"] = "undecided"
            else:
                if final_label_id in ["hatespeech", "offensive"]:
                    final_label_id = "toxic"
                else:
                    final_label_id = "non-toxic"
                temp["final_label"] = final_label_id

        else:
            if final_label.count(final_label_id) == 1:
                temp["final_label"] = "undecided"
            else:
                temp["final_label"] = final_label_id

        dict_data.append(temp)
    temp_read = pd.DataFrame(dict_data)
    return temp_read


def returnMask(row, params, tokenizer):

    text_tokens = row["text"]

    # a very rare corner case
    if len(text_tokens) == 0:
        text_tokens = ["dummy"]
        print("length of text ==0")
    #####

    mask_all = row["rationales"]
    mask_all_temp = mask_all
    count_temp = 0
    while len(mask_all_temp) != 3:
        mask_all_temp.append([0] * len(text_tokens))

    word_mask_all = []
    word_tokens_all = []

    for mask in mask_all_temp:
        if mask[0] == -1:
            mask = [0] * len(mask)

        list_pos = []
        mask_pos = []

        flag = 0
        for i in range(0, len(mask)):
            if i == 0 and mask[i] == 0:
                list_pos.append(0)
                mask_pos.append(0)

            if flag == 0 and mask[i] == 1:
                mask_pos.append(1)
                list_pos.append(i)
                flag = 1

            elif flag == 1 and mask[i] == 0:
                flag = 0
                mask_pos.append(0)
                list_pos.append(i)
        if list_pos[-1] != len(mask):
            list_pos.append(len(mask))
            mask_pos.append(0)
        string_parts = []
        for i in range(len(list_pos) - 1):
            string_parts.append(text_tokens[list_pos[i] : list_pos[i + 1]])

        if params["bert_tokens"]:
            word_tokens = [101]
            word_mask = [0]
        else:
            word_tokens = []
            word_mask = []

        for i in range(0, len(string_parts)):
            tokens = ek_extra_preprocess(" ".join(string_parts[i]), params, tokenizer)
            masks = [mask_pos[i]] * len(tokens)
            word_tokens += tokens
            word_mask += masks

        if params["bert_tokens"]:
            # always post truncation
            word_tokens = word_tokens[0 : (int(params["max_length"]) - 2)]
            word_mask = word_mask[0 : (int(params["max_length"]) - 2)]
            word_tokens.append(102)
            word_mask.append(0)

        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)

    #     for k in range(0,len(mask_all)):
    #          if(mask_all[k][0]==-1):
    #             word_mask_all[k] = [-1]*len(word_mask_all[k])
    if len(mask_all) == 0:
        word_mask_all = []
    else:
        word_mask_all = word_mask_all[0 : len(mask_all)]
    return word_tokens_all[0], word_mask_all


text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=["url", "email", "percent", "money", "phone", "user", "time", "date", "number"],
    # terms that will be annotated
    fix_html=True,  # fix HTML tokens
    annotate={"hashtag", "allcaps", "elongated", "repeated", "emphasis", "censored"},
    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",
    # corpus from which the word statistics are going to be used
    # for spell correction
    # corrector="twitter",
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons],
)


def ek_extra_preprocess(text, params, tokenizer):
    remove_words = [
        "<allcaps>",
        "</allcaps>",
        "<hashtag>",
        "</hashtag>",
        "<elongated>",
        "<emphasis>",
        "<repeated>",
        "'",
        "s",
    ]
    word_list = text_processor.pre_process_doc(text)
    if params["include_special"]:
        pass
    else:
        word_list = list(filter(lambda a: a not in remove_words, word_list))
    if params["bert_tokens"]:
        sent = " ".join(word_list)
        sent = re.sub(r"[<\*>]", " ", sent)
        sub_word_list = custom_tokenize(sent, tokenizer)
        return sub_word_list
    else:
        word_list = [token for token in word_list if token not in string.punctuation]
        return word_list


# Bert tokenizer


def custom_tokenize(sent, tokenizer, max_length=512):
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    try:

        encoded_sent = tokenizer.encode(
            sent,  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            # max_length = max_length,
            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.

    except ValueError:
        encoded_sent = tokenizer.encode(
            " ",  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,
        )
        # decide what to later

    return encoded_sent

def get_final_dict_with_rational(params, test_data=None, topk=5):
    final_list_dict = None
    test_data_path = test_data
    list_dict_org, test_data = standaloneEval_with_rational(
        params, extra_data_path=test_data, topk=topk)
    test_data_with_rational = convert_data(
        test_data, params, list_dict_org, rational_present=True, topk=topk)
    list_dict_with_rational, _ = standaloneEval_with_rational(
        params, test_data=test_data_with_rational, topk=topk, use_ext_df=True)
    test_data_without_rational = convert_data(
        test_data, params, list_dict_org, rational_present=False, topk=topk)
    list_dict_without_rational, _ = standaloneEval_with_rational(
        params, test_data=test_data_without_rational, topk=topk, use_ext_df=True)
    final_list_dict = []
    for ele1, ele2, ele3 in zip(list_dict_org, list_dict_with_rational, list_dict_without_rational):
        ele1['sufficiency_classification_scores'] = ele2['classification_scores']
        ele1['comprehensiveness_classification_scores'] = ele3['classification_scores']
        final_list_dict.append(ele1)

    # For real rationales data
    real_list_dict_org, test_data = real_standaloneEval_with_rational(
        params, extra_data_path=test_data_path, topk=topk)
    real_test_data_with_rational = convert_data(
        test_data, params, real_list_dict_org, rational_present=True, topk=topk)
    real_list_dict_with_rational, _ = real_standaloneEval_with_rational(
        params, test_data=real_test_data_with_rational, topk=topk, use_ext_df=True)
    real_test_data_without_rational = convert_data(
        test_data, params, real_list_dict_org, rational_present=False, topk=topk)
    real_list_dict_without_rational, _ = real_standaloneEval_with_rational(
        params, test_data=real_test_data_without_rational, topk=topk, use_ext_df=True)
    real_final_list_dict = []
    for ele1, ele2, ele3 in zip(real_list_dict_org, real_list_dict_with_rational, real_list_dict_without_rational):
        ele1['sufficiency_classification_scores'] = ele2['classification_scores']
        ele1['comprehensiveness_classification_scores'] = ele3['classification_scores']
        real_final_list_dict.append(ele1)

    return final_list_dict, real_final_list_dict

def standaloneEval_with_rational(params, test_data=None, extra_data_path=None, topk=2, use_ext_df=False):
    #     device = torch.device("cpu")
    if torch.cuda.is_available() and params['device'] == 'cuda':
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        deviceID = get_gpu(params)
        torch.cuda.set_device(deviceID[0])
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")

    embeddings = None
    if(params['bert_tokens']):
        train, val, test = createDatasetSplit(params)
        vocab_own = None
        vocab_size = 0
        padding_idx = 0
    else:
        train, val, test, vocab_own = createDatasetSplit(params)
        params['embed_size'] = vocab_own.embeddings.shape[1]
        params['vocab_size'] = vocab_own.embeddings.shape[0]
        embeddings = vocab_own.embeddings
    if(params['auto_weights']):
        y_test = [ele[2] for ele in test]
        encoder = LabelEncoder()
        encoder.classes_ = np.load('Data/classes_two.npy', allow_pickle=True)
        params['weights'] = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_test), y=y_test).astype('float32')
    if(extra_data_path != None):
        params_dash = {}
        params_dash['num_classes'] = 2

        params_dash['data_file'] = extra_data_path
        params_dash['data_file'] = 'Data/new_dataset.json'

        params_dash['class_names'] = dict_data_folder[str(
            params['num_classes'])]['class_label']
        temp_read = get_annotated_data(params_dash)

        # with open('Data/post_id_divisions.json', 'r') as fp:
        with open('Data/new_post_id_divisions.json', 'r') as fp:

            post_id_dict = json.load(fp)
        # temp_read = temp_read[temp_read['post_id'].isin(post_id_dict['test']) & (
        #     temp_read['final_label'].isin(['hatespeech', 'offensive']))]
        temp_read = temp_read[temp_read['post_id'].isin(post_id_dict['test']) & (
            temp_read['final_label'].isin(['toxic', 'non-toxic']))]
        test_data = get_test_data(temp_read, params, message='text')
        test_extra = encodeData(test_data, vocab_own, params)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    elif(use_ext_df):
        test_extra = encodeData(test_data, vocab_own, params)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    else:
        test_dataloader = combine_features(test, params, is_train=False)

    model = select_model(params, embeddings)
    if(params['bert_tokens'] == False):
        model = load_model(model, params)
    if(params["device"] == 'cuda'):
        model.cuda()
    model.eval()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables
    if((extra_data_path != None) or (use_ext_df == True)):
        post_id_all = list(test_data['Post_id'])
    else:
        post_id_all = list(test['Post_id'])

    print("Running eval on test data...")
    t0 = time.time()
    true_labels = []
    pred_labels = []
    logits_all = []
    attention_all = []
    input_mask_all = []

    input_ids_all = []
    ids_list = list(test_data['Post_id'])

    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        # model.zero_grad()
        outputs = model(b_input_ids,
                        attention_vals=b_att_val,
                        attention_mask=b_input_mask,
                        labels=None, device=device)
#         m = nn.Softmax(dim=1)
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.detach().cpu().numpy()

        if(params['bert_tokens']):
            attention_vectors = np.mean(
                outputs[1][11][:, :, 0, :].detach().cpu().numpy(), axis=1)
        else:
            attention_vectors = outputs[1].detach().cpu().numpy()

        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)
        attention_all += list(attention_vectors)
        input_mask_all += list(batch[2].detach().cpu().numpy())

        input_ids_all += list(b_input_ids.detach().cpu().numpy())

    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    if(use_ext_df == False):
        testf1 = f1_score(true_labels, pred_labels, average='macro')
        testacc = accuracy_score(true_labels, pred_labels)
        #testrocauc=roc_auc_score(true_labels, logits_all_final,multi_class='ovo',average='macro')
        testprecision = precision_score(
            true_labels, pred_labels, average='macro')
        testrecall = recall_score(true_labels, pred_labels, average='macro')

        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.3f}".format(testacc))
        print(" Fscore: {0:.3f}".format(testf1))
        print(" Precision: {0:.3f}".format(testprecision))
        print(" Recall: {0:.3f}".format(testrecall))
        #print(" Roc Auc: {0:.3f}".format(testrocauc))
        print(" Test took: {:}".format(format_time(time.time() - t0)))

    attention_vector_final = []
    input_ids_final = []
    for x, y, z in zip(attention_all, input_mask_all, input_ids_all):
        temp = []
        temp_z = []
        for x_ele, y_ele, z_ele in zip(x, y, z):
            if(y_ele == 1):
                temp.append(x_ele)
                temp_z.append(z_ele)
        attention_vector_final.append(temp)
        input_ids_final.append(temp_z)

# attention
# =======================================================================================

# predicted rationales

# =======================================================================================

    list_dict = []
    for post_id, attention, logits, pred, ground_truth in zip(post_id_all, attention_vector_final, logits_all_final, pred_labels, true_labels):
        #         if(ground_truth==1):
        #             continue
        temp = {}
        encoder = LabelEncoder()
        encoder.classes_ = np.load('Data/classes_two.npy', allow_pickle=True)
        pred_label = encoder.inverse_transform([pred])[0]
        ground_label = encoder.inverse_transform([ground_truth])[0]
        temp["annotation_id"] = post_id
        temp["classification"] = pred_label
        # temp["classification_scores"] = {
        #     "hatespeech": logits[0], "normal": logits[1], "offensive": logits[2]}
        temp["classification_scores"] = {
            "non-toxic": logits[0], "toxic": logits[1]}

        topk_indicies = sorted(range(len(attention)),
                               key=lambda i: attention[i])[-topk:]

        temp_hard_rationales = []
        for ind in topk_indicies:
            temp_hard_rationales.append(
                {'end_token': ind+1, 'start_token': ind})

        temp["rationales"] = [{"docid": post_id,
                               "hard_rationale_predictions": temp_hard_rationales,
                              "soft_rationale_predictions": attention,
                               # "soft_sentence_predictions":[1.0],
                               "truth": ground_truth}]
        list_dict.append(temp)

    return list_dict, test_data

def real_standaloneEval_with_rational(params, test_data=None, extra_data_path=None, topk=2, use_ext_df=False):
    #     device = torch.device("cpu")
    if torch.cuda.is_available() and params['device'] == 'cuda':
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        deviceID = get_gpu(params)
        torch.cuda.set_device(deviceID[0])
    else:
        print('Since you dont want to use GPU, using the CPU instead.')
        device = torch.device("cpu")

    embeddings = None
    if(params['bert_tokens']):
        train, val, test = createDatasetSplit(params)
        vocab_own = None
        vocab_size = 0
        padding_idx = 0
    else:
        train, val, test, vocab_own = createDatasetSplit(params)
        params['embed_size'] = vocab_own.embeddings.shape[1]
        params['vocab_size'] = vocab_own.embeddings.shape[0]
        embeddings = vocab_own.embeddings
    if(params['auto_weights']):
        y_test = [ele[2] for ele in test]
        encoder = LabelEncoder()
        encoder.classes_ = np.load('Data/classes_two.npy', allow_pickle=True)
        params['weights'] = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_test), y=y_test).astype('float32')
    if(extra_data_path != None):
        params_dash = {}
        params_dash['num_classes'] = 2
        params_dash['data_file'] = extra_data_path
        params_dash['class_names'] = dict_data_folder[str(
            params['num_classes'])]['class_label']
        temp_read = get_annotated_data(params_dash)
        with open('Data/post_id_divisions.json', 'r') as fp:
            post_id_dict = json.load(fp)
        # temp_read = temp_read[temp_read['post_id'].isin(post_id_dict['test']) & (
        #     temp_read['final_label'].isin(['hatespeech', 'offensive']))]
        temp_read = temp_read[temp_read['post_id'].isin(post_id_dict['test']) & (
            temp_read['final_label'].isin(['toxic', 'non-toxic']))]
        test_data = get_test_data(temp_read, params, message='text')
        test_extra = encodeData(test_data, vocab_own, params)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    elif(use_ext_df):
        test_extra = encodeData(test_data, vocab_own, params)
        test_dataloader = combine_features(test_extra, params, is_train=False)
    else:
        test_dataloader = combine_features(test, params, is_train=False)

    model = select_model(params, embeddings)
    if(params['bert_tokens'] == False):
        model = load_model(model, params)
    if(params["device"] == 'cuda'):
        model.cuda()
    model.eval()
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    # Tracking variables
    if((extra_data_path != None) or (use_ext_df == True)):
        post_id_all = list(test_data['Post_id'])
    else:
        post_id_all = list(test['Post_id'])

    print("Running eval on test data...")
    t0 = time.time()
    true_labels = []
    pred_labels = []
    logits_all = []
    attention_all = []
    input_mask_all = []

    # Evaluate data for one epoch
    for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

        # `batch` contains three pytorch tensors:
        #   [0]: input ids
        #   [1]: attention vals
        #   [2]: attention mask
        #   [3]: labels
        b_input_ids = batch[0].to(device)
        b_att_val = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_labels = batch[3].to(device)

        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        # model.zero_grad()
        outputs = model(b_input_ids,
                        attention_vals=b_att_val,
                        attention_mask=b_input_mask,
                        labels=None, device=device)
#         m = nn.Softmax(dim=1)
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.detach().cpu().numpy()

        if(params['bert_tokens']):
            attention_vectors = np.mean(
                outputs[1][11][:, :, 0, :].detach().cpu().numpy(), axis=1)
        else:
            attention_vectors = outputs[1].detach().cpu().numpy()

        # Calculate the accuracy for this batch of test sentences.
        # Accumulate the total accuracy.
        pred_labels += list(np.argmax(logits, axis=1).flatten())
        true_labels += list(label_ids.flatten())
        logits_all += list(logits)
        attention_all += list(attention_vectors)
        input_mask_all += list(batch[2].detach().cpu().numpy())

    logits_all_final = []
    for logits in logits_all:
        logits_all_final.append(softmax(logits))

    if(use_ext_df == False):
        testf1 = f1_score(true_labels, pred_labels, average='macro')
        testacc = accuracy_score(true_labels, pred_labels)
        #testrocauc=roc_auc_score(true_labels, logits_all_final,multi_class='ovo',average='macro')
        testprecision = precision_score(
            true_labels, pred_labels, average='macro')
        testrecall = recall_score(true_labels, pred_labels, average='macro')

        # Report the final accuracy for this validation run.
        print(" Accuracy: {0:.3f}".format(testacc))
        print(" Fscore: {0:.3f}".format(testf1))
        print(" Precision: {0:.3f}".format(testprecision))
        print(" Recall: {0:.3f}".format(testrecall))
        #print(" Roc Auc: {0:.3f}".format(testrocauc))
        print(" Test took: {:}".format(format_time(time.time() - t0)))

    attention_vector_final = []
    for x, y in zip(attention_all, input_mask_all):
        temp = []
        for x_ele, y_ele in zip(x, y):
            if(y_ele == 1):
                temp.append(x_ele)
        attention_vector_final.append(temp)

# =======================================================================================

# real rationales

# =======================================================================================

    real_list_dict = []
    real_attention_vector_final = [list(test[1]) for test in test_extra]
    for post_id, attention, logits, pred, ground_truth in zip(post_id_all, real_attention_vector_final, logits_all_final, pred_labels, true_labels):
        #         if(ground_truth==1):
        #             continue
        real_temp = {}
        encoder = LabelEncoder()
        encoder.classes_ = np.load('Data/classes_two.npy', allow_pickle=True)
        pred_label = encoder.inverse_transform([pred])[0]
        ground_label = encoder.inverse_transform([ground_truth])[0]
        real_temp["annotation_id"] = post_id
        real_temp["classification"] = pred_label
        # real_temp["classification_scores"] = {
        #     "hatespeech": logits[0], "normal": logits[1], "offensive": logits[2]}
        real_temp["classification_scores"] = {
            "non-toxic": logits[0], "toxic": logits[1]}

        topk_indicies = sorted(range(len(attention)),
                               key=lambda i: attention[i])[-topk:]

        real_temp_hard_rationales = []
        for ind in topk_indicies:
            real_temp_hard_rationales.append(
                {'end_token': ind+1, 'start_token': ind})

        real_temp["rationales"] = [{"docid": post_id,
                                    "hard_rationale_predictions": real_temp_hard_rationales,
                                    "soft_rationale_predictions": attention,
                                    # "soft_sentence_predictions":[1.0],
                                    "truth": ground_truth}]
        real_list_dict.append(real_temp)

    return real_list_dict, test_data