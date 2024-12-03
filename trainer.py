import ast
import json
import os
from os import path
from datetime import datetime, timedelta
import time

import wandb
import torch
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer
import pickle

import more_itertools as mit

import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from src.utils.utils import train_model, Eval_phase, returnMask, get_final_dict_with_rational
from code.AudioHateXplain.src.data.datsetSplitter import createDatasetSplit, encodeData
from code.AudioHateXplain.src.data.dataLoader import combine_features
from Preprocess.attentionCal import aggregate_attention
from Preprocess.dataCollect import get_annotated_data, convert_data, get_test_data
from Models.bertModels import SC_weighted_BERT

from eraserbenchmark.rationale_benchmark.metrics import calculate_rationales
from testing_with_rational import save_pred_real_rationales

from Models.utils import masked_cross_entropy, softmax, return_params, fix_the_random, format_time, get_gpu, return_params, load_model

PATH = os.path.dirname(os.path.realpath(__file__))
os.chdir(PATH)

model_dict_params = {
    'bert': 'best_model_json/bestModel_bert_base_uncased_Attn_train_FALSE.json',
    'bert_supervised': 'best_model_json/BERT_test_whisperX-tiny_audioHateXplain_train_audioHateXplainGold.json', # 🔥 Modify this config for changing model
    'birnn': 'best_model_json/bestModel_birnn.json',
    'cnngru': 'best_model_json/bestModel_cnn_gru.json',
    'birnn_att': 'best_model_json/bestModel_birnnatt.json',
    'birnn_scrat': 'best_model_json/bestModel_birnnscrat.json'
}

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

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class HateSpeechTrainer(object):
    def __init__(self, param, model_type,
                 train_dataset, val_dataset, test_dataset,
                 model, tokenizer):
        self.params = None
        self.device = None
        self.model_dict_params = {
                                    'bert_supervised': f"best_model_json/{model_type}.json", # 🔥 Modify this config for changing model
                                }
        self.params = param
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.tokenizer = tokenizer

    def train(self):
        # for att_lambda in [0.001,0.01,0.1,1,10,100]
        train_model(params=self.params,
                    device=self.device,
                    train=self.train_dataset,
                    val=self.val_dataset,
                    test=self.test_dataset,
                    model=self.model,
                    tokenizer=self.tokenizer
                    )

    def predict(self, test_dataset=None, threshold=None, wer_datapoint_graph=False):
        params = self.params
        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time + f"{params['model_name']}+ep={params['epochs']}+bs={params['batch_size']}"

        # TODO: wandb 계정이름 바꾸기
        wandb.init(
            project="IJCAI2024_HateSpeech",
            entity="jinmyeong",
            name=name,
            config={
                "learning_rate": params["learning_rate"],
                "epochs": params["epochs"],
                "batch_size": params["batch_size"],
            },
        )

        embeddings = None
        if params["bert_tokens"]:
            train, val, test = createDatasetSplit(params)
        else:
            train, val, test, vocab_own = createDatasetSplit(params)
            params["embed_size"] = vocab_own.embeddings.shape[1]
            params["vocab_size"] = vocab_own.embeddings.shape[0]
            embeddings = vocab_own.embeddings
        if params["auto_weights"]:
            y_test = [ele[2] for ele in test]
            #         print(y_test)
            encoder = LabelEncoder()
            encoder.classes_ = np.load(params["class_names"], allow_pickle=True)
            params["weights"] = class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(y_test), y=y_test
            ).astype("float32")
            # params['weights']=np.array([len(y_test)/y_test.count(encoder.classes_[0]),len(y_test)/y_test.count(encoder.classes_[1]),len(y_test)/y_test.count(encoder.classes_[2])]).astype('float32')

        print(params["weights"])
        # train_dataloader = combine_features(train, params, is_train=True)
        validation_dataloader = combine_features(val, params, is_train=False)
        test_dataloader = combine_features(test, params, is_train=False)

        val_id_list = [ele[3] for ele in val] 
        test_id_list = [ele[3] for ele in test] 

        
        params["path_files"] = params["save_model_path"]
        model = select_model(params, embeddings)
        
        # do only evaluation
        model.eval()
        if params["device"] == "cuda":
            model.cuda()

        # Set the seed value all over the place to make this reproducible.
        fix_the_random(seed_val=params["random_seed"])
        # Store the average loss after each epoch so we can plot them.
        
        # threshold for changing soft att into hard att
        params['att_threshold'] = threshold
        # find val distribution in different threshold

        # (val_fscore, val_accuracy, val_precision, val_recall, val_roc_auc, logits_all_final,
        # iou_all_data_all_label, iou_all_data_hate_label, iou_hate_data_all_label, 
        # iou_hate_data_hate_label, precisions_all_data, recalls_all_data, f1s_all_data,
        # precisions_hate_data, recalls_hate_data, f1s_hate_data) = Eval_phase(
        #     params, "val", model, validation_dataloader, params["device"], val_id_list
        # )


        (test_fscore, test_accuracy, test_precision, test_recall, test_roc_auc, logits_all_final,
        iou_all_data_all_label, iou_all_data_hate_label, iou_hate_data_all_label, 
        iou_hate_data_hate_label, precisions_all_data, recalls_all_data, f1s_all_data,
        precisions_hate_data, recalls_hate_data, f1s_hate_data) = Eval_phase(
            params, "test", model, test_dataloader, params["device"], test_id_list, wer_datapoint_graph=wer_datapoint_graph
        )

        # apply att threshold to val distribution
        # (test_fscore, test_accuracy, test_precision, test_recall, test_roc_auc, logits_all_final,
        # iou_all_data_all_label, iou_all_data_hate_label, iou_hate_data_all_label, 
        # iou_hate_data_hate_label, precisions_all_data, recalls_all_data, f1s_all_data,
        # precisions_hate_data, recalls_hate_data, f1s_hate_data) = Eval_phase(
        #     params, "val", model, validation_dataloader, params["device"], val_id_list
        # )

        wandb.log(
            {
                "test_accuracy": test_accuracy,
                "test_fscore": test_fscore,
                "test_recall": test_recall,
                "iou_all_data_all_label": iou_all_data_all_label,
                "iou_all_data_hate_label": iou_all_data_hate_label,
                "iou_hate_data_all_label": iou_hate_data_all_label,
                "iou_hate_data_hate_label": iou_hate_data_hate_label,
                "recalls_all_data": recalls_all_data,
                "f1s_all_data": f1s_all_data,
                "recalls_hate_data": recalls_hate_data,
                "f1s_hate_data": f1s_hate_data
            }
        )  
            
        print(f"Result Type: {self.model_dict_params['bert_supervised']}")
        print(f"test_accuracy: {test_accuracy}")
        print(f"test_fscore: {test_fscore}" )
        print(f"test_recall: {test_recall}")
        print(f"test_precision: {test_precision}")

        print(f"iou_hate_data_all_label: {iou_hate_data_all_label}")
        print(f"iou_all_data_all_label: {iou_all_data_all_label}")
        print(f"iou_hate_data_hate_label: {iou_hate_data_hate_label}")
        print(f"iou_all_data_hate_label: {iou_all_data_hate_label}")
        print(f"precisions_hate_data: {precisions_hate_data}")
        print(f"precisions_all_data: {precisions_all_data}")
        print(f"recalls_hate_data: {recalls_hate_data}")
        print(f"f1s_hate_data: {f1s_hate_data}")
        
        result_config = self.model_dict_params['bert_supervised']
        result = {"test_accuracy": test_accuracy,
                "test_fscore": test_fscore,
                "test_recall": test_recall,
                "test_precision": test_precision,
                "iou_all_data_all_label": iou_all_data_all_label,
                "iou_all_data_hate_label": iou_all_data_hate_label,
                "precisions_all_data": precisions_all_data,
                "recalls_hate_data": recalls_hate_data,
                "f1s_hate_data": f1s_hate_data}
        
        return result_config, result

    def get_test_dataset(self, path='Data/Total_data_bert_softmax_1_128_2'):
        with open(os.path.join(path, "test_data.pickle"), "rb") as f:
            X_test = pickle.load(f)
        return X_test

    def get_training_data(self, data, params, tokenizer):
        """input: data is a dataframe text ids attentions labels column only"""
        """output: training data in the columns post_id,text, attention and labels """

        majority = params["majority"]
        post_ids_list = []
        text_list = []
        attention_list = []
        label_list = []
        count = 0
        count_confused = 0
        print("total_data", len(data))
        for index, row in tqdm(data.iterrows(), total=len(data)):
            # print(params)
            text = row["text"]
            post_id = row["post_id"]

            annotation_list = [row["label1"], row["label2"], row["label3"]]
            annotation = row["final_label"]

            if annotation != "undecided":
                tokens_all, attention_masks = returnMask(row, params, tokenizer)
                attention_vector = aggregate_attention(attention_masks, row, params)
                attention_list.append(attention_vector)
                text_list.append(tokens_all)
                label_list.append(annotation)
                post_ids_list.append(post_id)
            else:
                count_confused += 1

        print("attention_error:", count)
        print("no_majority:", count_confused)
        # Calling DataFrame constructor after zipping
        # both lists, with columns specified
        training_data = pd.DataFrame(
            list(zip(post_ids_list, text_list, attention_list, label_list)),
            columns=["Post_id", "Text", "Attention", "Label"],
        )

        return training_data

    def encodeData(self, dataframe, vocab, params):
        tuple_new_data = []
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            if params["bert_tokens"]:
                tuple_new_data.append((row["Text"], row["Attention"], row["Label"]))
            else:
                list_token_id = []
                for word in row["Text"]:
                    try:
                        index = vocab.stoi[word]
                    except KeyError:
                        index = vocab.stoi["unk"]
                    list_token_id.append(index)
                tuple_new_data.append((list_token_id, row["Attention"], row["Label"]))
        return tuple_new_data

    
    def get_explanation(self):
        my_parser = argparse.ArgumentParser(description='Which model to use')

        # Add the arguments
        my_parser.add_argument('--model_to_use',
                            #    metavar='--model_to_use',
                            type=str,
                            default='bert_supervised',
                            required=False,
                            help='model to use for evaluation')

        my_parser.add_argument('--attention_lambda',
                            #    metavar='--attention_lambda',
                            type=str,
                            default='100',
                            required=False,
                            help='required to assign the contribution of the atention loss')

        args = my_parser.parse_args()

        model_to_use = args.model_to_use

        params = return_params(
            self.model_dict_params[model_to_use], float(args.attention_lambda))
        params['path_files'] = 'Saved/bert-base-uncased_11_6_2_0.001'
        params['variance'] = 1
        params['num_classes'] = 2
        params['device'] = 'cpu'
        fix_the_random(seed_val=params['random_seed'])
        params['class_names'] = dict_data_folder[str(
            params['num_classes'])]['class_label']
        params['data_file'] = dict_data_folder[str(
            params['num_classes'])]['data_file']
        # params['data_file'] = None
        # test_data=get_test_data(temp_read,params,message='text')
        final_list_dict, real_final_list_dict = get_final_dict_with_rational(
            params, params['data_file'], topk=5)

        path_name = self.model_dict_params[model_to_use]
        path_name_explanation = 'explanations_dicts/' + \
            path_name.split('/')[1].split('.')[0]+'_' + \
            str(params['att_lambda'])+'_explanation_top5.json'
        with open(path_name_explanation, 'w') as fp:
            fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder)
                    for i in final_list_dict))

        path_name_explanation = 'explanations_dicts/real_' + path_name.split('/')[1].split('.')[0]+'_' + \
            str(params['att_lambda'])+'_explanation_top5.json'
        # for real_rationales 'real_'+
        with open(path_name_explanation, 'w') as fp:
            fp.write('\n'.join(json.dumps(i, cls=NumpyEncoder)
                    for i in real_final_list_dict))       

    def get_pred_soft_token_rationales(self, model_path='Saved/bert-base-uncased_11_6_2_0.001'):
        my_parser = argparse.ArgumentParser(description='Which model to use')

        # Add the arguments
        my_parser.add_argument('--model_to_use',
                            #    metavar='--model_to_use',
                            type=str,
                            default='bert_supervised',
                            required=False,
                            help='model to use for evaluation')

        my_parser.add_argument('--attention_lambda',
                            #    metavar='--attention_lambda',
                            type=str,
                            default='100',
                            required=False,
                            help='required to assign the contribution of the atention loss')

        args = my_parser.parse_args()

        model_to_use = args.model_to_use

        params = return_params(
            self.model_dict_params[model_to_use], float(args.attention_lambda))
        params['path_files'] = model_path
        params['variance'] = 1
        params['num_classes'] = 2
        params['device'] = 'cpu'
        fix_the_random(seed_val=params['random_seed'])
        params['class_names'] = dict_data_folder[str(
            params['num_classes'])]['class_label']
        params['data_file'] = 'Data/new_dataset.json'
        
        if torch.cuda.is_available() and params['device'] == 'cuda':
            # Tell PyTorch to use the GPU.
            device = torch.device("cuda")
            deviceID = get_gpu(params)
            torch.cuda.set_device(deviceID[0])
        else:
            print('Since you dont want to use GPU, using the CPU instead.')
            device = torch.device("cpu")

        embeddings = None
        # if(params['bert_tokens']):
        #     train, val, test = createDatasetSplit(params)
        #     vocab_own = None
        #     vocab_size = 0
        #     padding_idx = 0
        # if(params['auto_weights']):
        #     y_test = [ele[2] for ele in test]
        #     encoder = LabelEncoder()
        #     encoder.classes_ = np.load('Data/classes_two.npy', allow_pickle=True)
        #     params['weights'] = class_weight.compute_class_weight(
        #         class_weight='balanced', classes=np.unique(y_test), y=y_test).astype('float32')
        vocab_own = None    
        extra_data_path='Data/new_dataset.json'

        if(extra_data_path != None):
            params_dash = {}
            params_dash['num_classes'] = 2

            params_dash['data_file'] = extra_data_path

            params_dash['class_names'] = dict_data_folder[str(
                params['num_classes'])]['class_label']
            temp_read = get_annotated_data(params_dash)

            with open('Data/new_post_id_divisions.json', 'r') as fp:
                post_id_dict = json.load(fp)
 
            temp_read = temp_read[temp_read['post_id'].isin(post_id_dict['test']) & (
                temp_read['final_label'].isin(['toxic', 'non-toxic']))]
            test_data = get_test_data(temp_read, params, message='text')
            test_extra = encodeData(test_data, vocab_own, params)
            test_dataloader = combine_features(test_extra, params, is_train=False)

        model = select_model(params, embeddings)
        if(params['bert_tokens'] == False):
            model = load_model(model, params)
        if(params["device"] == 'cuda'):
            model.cuda()
        model.eval()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        # Tracking variables
        use_ext_df=False
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
        for idx, id in enumerate(ids_list):
            pred_soft_attention[id] = {'pred_soft_attention': ([float(att) for att in attention_vector_final[idx]]),
                                       'input_ids': [float(input_id) for input_id in input_ids_final[idx]],
                                       }
        file_path = '/home/jinmyeong/code/hts/classifier/Result/bert_cls+att/pred_soft_attention.json'
        with open(file_path, "w") as f:
            json.dump(pred_soft_attention, f)     

        return pred_soft_attention
    def get_pred_hard_token_rationales(self, pred_soft_attention):
        for key in pred_soft_attention:
            soft_attention = pred_soft_attention[key]['pred_soft_attention']
            mean_attention = np.mean(soft_attention)

            pred_soft_attention[key]['pred_hard_attention'] = [1 if att >= mean_attention else 0 for att in soft_attention ]
            # TODO: majority filter

        return pred_soft_attention
    
    def get_pred_hard_word_rationales(self, pred_hard_attention, tokenizer):
        file_path = '/home/jinmyeong/code/hts/classifier/Data/new_dataset.json'
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

    def get_pred_timestamp(self, pred_hard_word_rationales):
        with open('/home/jinmyeong/code/hts/ASR/Data/ASR/whisperX_ASR_transcription_v4-HateSpeech-all-multispeaker.json', 'r') as fp:
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
    
    def get_rationales_predict(self):
        calculate_rationales()

