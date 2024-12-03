import os
import pickle
import time
from collections import deque
from datetime import datetime, timedelta
import json
import numpy as np
import sklearn.metrics as metrics
import torch
from datasets import Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, EarlyStoppingCallback, Trainer, TrainingArguments
import math

import wandb
from src.utils import build_utt_win_dataset, build_utt_win_dataset_no_SEP, metrics_test


# Create the TrainerFactory class
class TrainerFactory:
    def __init__(self):
        # This dictionary maps the string identifiers to the trainer classes.
        self.trainers = {
            "chatBERTTrainer": ChatBERTTrainer,
            "chatBERTTrainer_seg_ann": ChatBERTTrainer_seg_ann,
            "chatBERTTrainer_stage_ann": ChatBERTTrainer_stage_ann,
            "segBERTTrainer": SegBERTTrainer,
            "segBERTTrainer_chat_ann": SegBERTTrainer_chat_ann,
            "segBERTTrainer_stage_ann": SegBERTTrainer_stage_ann,
            "segBERTTrainer_stage_ann_speaker": SegBERTTrainer_stage_ann_speaker,
            "segBERTTrainer_allStage_ann_speaker": SegBERTTrainer_allStage_ann_speaker,
            "segBERTTrainer_stage_ann_chat_seg": SegBERTTrainer_stage_ann_chat_seg,
            # Add other mappings here as needed.
        }

    def get_trainer(self, trainer_type):
        # Get the class from the dictionary and instantiate it.
        trainer_class = self.trainers.get(trainer_type)
        if trainer_class:
            return trainer_class
        raise ValueError(f"Unknown trainer type: {trainer_type}")


class ChatBERTTrainer(Trainer):
    @torch.no_grad()
    def skeptical_predict(self, data_args, test_dataset: Dataset, device="cuda"):
        window_size = 50
        skeptical = 10
        pred_label_list = []
        early_stop_len_utt_list = []
        risk_fail = []
        normal_fail = 0
        sents_len = []  # 각 대화를 구성하는 전체 문장 수

        doc_labels = test_dataset["label"]

        self.tokenizer.truncation_side = "left"
        self.model.eval()

        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time
        wandb.init(
            project=data_args.task_name,
            entity="jinmyeong",
            name=name,
        )

        # 1. iterate test datset

        for test_datapoint in tqdm(test_dataset):
            label = test_datapoint["label"]
            queue = []
            pred_riskness = "normal"
            early_stop_len_utt = 0

            if label == 1:  # risk 인 대화의 발화 문장 개수
                # label == 'risk' 가 되어야하는거 아님?
                sents_len.append(len(test_datapoint["text"]))

            win_test_datapoint = build_utt_win_dataset_no_SEP(test_datapoint)
            win_test_dataloader = DataLoader(
                dataset=win_test_datapoint, batch_size=self.args.per_device_eval_batch_size, shuffle=False
            )  # , collate_fn=self.collate_fn)
            prob_list = torch.Tensor([])

            for idx, batch in enumerate(win_test_dataloader):
                encoded_batch = self.tokenizer(
                    batch["text"], padding="max_length", truncation=True, return_tensors="pt"
                )
                encoded_batch.to(device=self.model.device)
                probs = self.model(
                    input_ids=encoded_batch["input_ids"],
                    token_type_ids=encoded_batch["token_type_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                )
                prob_list = torch.cat((prob_list, probs.logits.cpu()), 0)
            for idx, prob in enumerate(prob_list):
                pred_riskness = "normal"
                early_stop_len_utt += 1
                pred_label = "normal" if np.argmax(prob, axis=0) == 0 else "risk"

                if len(queue) < 10:
                    queue.append(pred_label)
                else:
                    queue.pop(0)
                    queue.append(pred_label)

                risk_num = self.get_risk_num(queue)
                if risk_num >= 5:
                    pred_riskness = "risk"
                    break
            # determine risk or not
            pred_datapoint_label = 1 if pred_riskness == "risk" else 0
            early_stop_len_utt_list.append(early_stop_len_utt)

            pred_label_list.append(pred_datapoint_label)

            ############## Error Analysis ##############
            # true risk and pred normal
            if label == 1 and pred_datapoint_label == 0:
                risk_fail.append(1)
            # true normal and pred risk
            elif label == 0 and pred_datapoint_label == 1:
                normal_fail += 1
            ############################################

        # TODO: Evaluate recall, precision, F1, and Latency F1
        doc_trues = torch.tensor(doc_labels)  # data 개수 x 1
        label_preds = torch.tensor(pred_label_list)
        print("Detect 하지 못한 risk : \n", risk_fail)
        print("\nNum(risk_fail) : ", len(risk_fail))
        print("Num(normal_fail) : ", normal_fail)

        ############## Performace #################
        # classification_performace = metrics.classification_report(doc_trues, label_preds, target_names=['risk', 'normal'])
        classification_performance = metrics.classification_report(doc_trues, label_preds)
        micro_F1 = metrics.f1_score(doc_trues, label_preds, average="micro")
        risk_micro_F1 = metrics.f1_score(doc_trues, label_preds, pos_label=1, average="binary")

        p = np.log(3) * (1 / (np.median(sents_len) - 1))
        MESSAGE_WITH_HALF_PENALTY = 90
        # p = np.log(3)/(MESSAGE_WITH_HALF_PENALTY-1)

        Latency_F1 = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p) * metrics.f1_score(
            doc_trues, label_preds, pos_label=1, average="binary"
        )
        # metrics.f1_score(doc_trues, label_preds, pos_label=0, average='binary')
        # ERDE = metrics_test.early_risk_detection_error(
        #     args.error_score, args.deadline, doc_trues, label_preds, early_stop_len_utt_list
        # )

        wandb.log(
            {
                "micro_F1": micro_F1,
                "Latency_F1": Latency_F1,
                # "ERDE": ERDE,
                "speed": metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p),
            }
        )

        return (
            classification_performance,
            micro_F1,
            risk_micro_F1,
            Latency_F1,
            # ERDE,
            early_stop_len_utt_list,
            doc_trues,
            label_preds,
        )

    @torch.no_grad()
    def predict_chats(self):
        dataloader = self.get_eval_dataloader(eval_dataset=None)
        for idx, batch in tqdm(enumerate(dataloader)):
            batch = {k: v.to(device=self.model.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            outputs_prob = torch.nn.functional.softmax(outputs.logits, dim=1)

        return outputs_prob

    def get_risk_num(self, queue):
        num_risk = len([1 for label in queue if label == "risk"])
        return num_risk


class ChatBERTTrainer_seg_ann(Trainer):
    @torch.no_grad()
    def skeptical_predict(self, data_args, test_dataset: Dataset, device="cuda"):
        window_size = 50
        skeptical = 10
        pred_label_list = []
        early_stop_len_utt_list = []
        risk_fail = []
        normal_fail = 0
        sents_len = []  # 각 대화를 구성하는 전체 문장 수
        # if type(test_dataset['labels'][0]) == int:
        #     doc_labels = test_dataset['labels']
        # else:
        #     doc_labels = [1 if real_label[0][0] == 'normal' else 0 for real_label in test_dataset['labels']]
        doc_labels = []

        self.tokenizer.truncation_side = "left"
        self.model.eval()

        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time
        wandb.init(
            project=data_args.task_name,
            entity="jinmyeong",
            name=name,
        )
        # 1. iterate test datset
        for test_datapoint in tqdm(test_dataset):
            utt_list = test_datapoint["text"]
            label = test_datapoint["label"]
            # label = 1 if test_datapoint['labels'][0][0] == 'risk' else 0
            queue = []
            pred_riskness = "normal"
            early_stop_len_utt = 0

            if label == 1:  # risk 인 대화의 발화 문장 개수
                # label == 'risk' 가 되어야하는거 아님?
                sents_len.append(len(utt_list))

            win_test_datapoint = build_utt_win_dataset_no_SEP(test_datapoint)
            win_test_dataloader = DataLoader(
                dataset=win_test_datapoint, batch_size=self.args.per_device_eval_batch_size, shuffle=False
            )  # , collate_fn=self.collate_fn)
            prob_list = torch.Tensor([])

            for idx, batch in enumerate(win_test_dataloader):
                encoded_batch = self.tokenizer(
                    batch["text"], padding="max_length", truncation=True, return_tensors="pt"
                )
                encoded_batch.to(device=self.model.device)
                probs = self.model(
                    input_ids=encoded_batch["input_ids"],
                    token_type_ids=encoded_batch["token_type_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                )
                prob_list = torch.cat((prob_list, probs.logits.cpu()), 0)
            for idx, prob in enumerate(prob_list):
                pred_riskness = "normal"
                early_stop_len_utt += 1
                pred_label = "normal" if np.argmax(prob, axis=0) == 0 else "risk"

                # for idx, utt in enumerate(utt_list):
                #     pred_riskness = "normal"
                #     early_stop_len_utt += 1

                #     window = utt_list[max(0, idx + 1 - window_size) : idx + 1]
                #     text = " ".join(window)
                #     encoded_text = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

                #     input_ids = encoded_text["input_ids"].to(device)
                #     token_type_ids = encoded_text["token_type_ids"].to(device)
                #     attention_mask = encoded_text["attention_mask"].to(device)

                #     prob = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                #     pred_label = "normal" if np.argmax(prob.logits.cpu(), axis=1) == 0 else "risk"
                #     # pred_label = 'normal' if np.argmax(prob.logits.cpu(), axis=1) == 0 else 'risk'#

                if len(queue) < 10:
                    queue.append(pred_label)
                else:
                    queue.pop(0)
                    queue.append(pred_label)

                risk_num = self.get_risk_num(queue)
                if risk_num >= 5:  # 5
                    pred_riskness = "risk"
                    break

            # determine risk or not
            pred_datapoint_label = 1 if pred_riskness == "risk" else 0
            true_label = 1 if test_datapoint["utt_annotation"][idx] == 1 else 0
            doc_labels.append(true_label)

            early_stop_len_utt_list.append(early_stop_len_utt)

            pred_label_list.append(pred_datapoint_label)

            ############## Error Analysis ##############
            # true risk and pred normal
            if label == 1 and pred_datapoint_label == 0:
                risk_fail.append(1)
            # true normal and pred risk
            elif label == 0 and pred_datapoint_label == 1:
                normal_fail += 1
            ############################################

        # TODO: Evaluate recall, precision, F1, and Latency F1
        doc_trues = torch.tensor(doc_labels)  # data 개수 x 1
        label_preds = torch.tensor(pred_label_list)
        print("Detect 하지 못한 risk : \n", risk_fail)
        print("\nNum(risk_fail) : ", len(risk_fail))
        print("Num(normal_fail) : ", normal_fail)

        ############## Performace #################
        # classification_performace = metrics.classification_report(doc_trues, label_preds, target_names=['risk', 'normal'])
        classification_performance = metrics.classification_report(doc_trues, label_preds)
        micro_F1 = metrics.f1_score(doc_trues, label_preds, average="micro")
        risk_micro_F1 = metrics.f1_score(doc_trues, label_preds, pos_label=1, average="binary")

        p = np.log(3) * (1 / (np.median(sents_len) - 1))
        MESSAGE_WITH_HALF_PENALTY = 90
        # p = np.log(3)/(MESSAGE_WITH_HALF_PENALTY-1)

        Latency_F1 = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p) * metrics.f1_score(
            doc_trues, label_preds, pos_label=1, average="binary"
        )
        # metrics.f1_score(doc_trues, label_preds, pos_label=0, average='binary')
        # ERDE = metrics_test.early_risk_detection_error(
        #     args.error_score, args.deadline, doc_trues, label_preds, early_stop_len_utt_list
        # )

        wandb.log(
            {
                "micro_F1": micro_F1,
                "Latency_F1": Latency_F1,
                # "ERDE": ERDE,
                "speed": metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p),
            }
        )

        return (
            classification_performance,
            micro_F1,
            risk_micro_F1,
            Latency_F1,
            # ERDE,
            early_stop_len_utt_list,
            doc_trues,
            label_preds,
        )

    def get_risk_num(self, queue):
        num_risk = len([1 for label in queue if label == "risk"])
        return num_risk


class SegBERTTrainer(Trainer):
    @torch.no_grad()
    def skeptical_predict(self, data_args, test_dataset: Dataset, device="cuda"):
        window_size = 50
        skeptical = 10
        pred_label_list = []
        early_stop_len_utt_list = []
        risk_fail = []
        normal_fail = 0
        sents_len = []  # 각 대화를 구성하는 전체 문장 수
        # if type(test_dataset['labels'][0]) == int:
        #     doc_labels = test_dataset['labels']
        # else:
        #     doc_labels = [1 if real_label[0][0] == 'normal' else 0 for real_label in test_dataset['labels']]
        doc_labels = []

        self.tokenizer.truncation_side = "left"
        self.model.eval()

        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time
        wandb.init(
            project=data_args.task_name,
            entity="jinmyeong",
            name=name,
        )
        # 1. iterate test datset
        for test_datapoint in tqdm(test_dataset):
            utt_list = test_datapoint["text"]
            # label = test_datapoint["label"]
            label = 1 if test_datapoint["label"] == "risk" else 0
            # label = 1 if test_datapoint['labels'][0][0] == 'risk' else 0
            queue = []
            pred_riskness = "normal"
            early_stop_len_utt = 0

            if label == 1:  # risk 인 대화의 발화 문장 개수
                # label == 'risk' 가 되어야하는거 아님?
                sents_len.append(len(utt_list))

            win_test_datapoint = build_utt_win_dataset(test_datapoint)
            win_test_dataloader = DataLoader(
                dataset=win_test_datapoint, batch_size=self.args.per_device_eval_batch_size, shuffle=False
            )  # , collate_fn=self.collate_fn)
            prob_list = torch.Tensor([])

            for idx, batch in enumerate(win_test_dataloader):
                encoded_batch = self.tokenizer(
                    batch["text"], padding="max_length", truncation=True, return_tensors="pt"
                )
                encoded_batch.to(device=self.model.device)
                probs = self.model(
                    input_ids=encoded_batch["input_ids"],
                    token_type_ids=encoded_batch["token_type_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                )
                prob_list = torch.cat((prob_list, probs.logits.cpu()), 0)
            for idx, prob in enumerate(prob_list):
                pred_riskness = "normal"
                early_stop_len_utt += 1
                pred_label = "normal" if np.argmax(prob, axis=0) == 0 else "risk"

                # for idx, utt in enumerate(utt_list):
                #     pred_riskness = "normal"
                #     early_stop_len_utt += 1

                #     window = utt_list[max(0, idx + 1 - window_size) : idx + 1]
                #     text = " ".join(window)
                #     encoded_text = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

                #     input_ids = encoded_text["input_ids"].to(device)
                #     token_type_ids = encoded_text["token_type_ids"].to(device)
                #     attention_mask = encoded_text["attention_mask"].to(device)

                #     prob = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                #     pred_label = "normal" if np.argmax(prob.logits.cpu(), axis=1) == 0 else "risk"
                #     # pred_label = 'normal' if np.argmax(prob.logits.cpu(), axis=1) == 0 else 'risk'#

                if len(queue) < 10:
                    queue.append(pred_label)
                else:
                    queue.pop(0)
                    queue.append(pred_label)

                risk_num = self.get_risk_num(queue)
                if risk_num >= 5:  # 5
                    pred_riskness = "risk"
                    break

            # determine risk or not
            pred_datapoint_label = 1 if pred_riskness == "risk" else 0
            true_label = 1 if test_datapoint["utt_annotation"][idx] == 1 else 0
            doc_labels.append(true_label)

            early_stop_len_utt_list.append(early_stop_len_utt)

            pred_label_list.append(pred_datapoint_label)

            ############## Error Analysis ##############

            # true risk and pred normal
            if label == 1 and pred_datapoint_label == 0:
                risk_fail.append(1)
            # true normal and pred risk
            elif label == 0 and pred_datapoint_label == 1:
                normal_fail += 1
            # true risk and pred risk
            elif label == 1 and pred_datapoint_label == 1:
                print()
            ############################################

        # TODO: Evaluate recall, precision, F1, and Latency F1
        doc_trues = torch.tensor(doc_labels)  # data 개수 x 1
        label_preds = torch.tensor(pred_label_list)
        print("Detect 하지 못한 risk : \n", risk_fail)
        print("\nNum(risk_fail) : ", len(risk_fail))
        print("Num(normal_fail) : ", normal_fail)

        ############## Performace #################
        # classification_performace = metrics.classification_report(doc_trues, label_preds, target_names=['risk', 'normal'])
        classification_performance = metrics.classification_report(doc_trues, label_preds)
        micro_F1 = metrics.f1_score(doc_trues, label_preds, average="micro")
        risk_micro_F1 = metrics.f1_score(doc_trues, label_preds, pos_label=1, average="binary")

        p = np.log(3) * (1 / (np.median(sents_len) - 1))
        MESSAGE_WITH_HALF_PENALTY = 90
        # p = np.log(3)/(MESSAGE_WITH_HALF_PENALTY-1)

        Latency_F1 = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p) * metrics.f1_score(
            doc_trues, label_preds, pos_label=1, average="binary"
        )
        # metrics.f1_score(doc_trues, label_preds, pos_label=0, average='binary')
        # ERDE = metrics_test.early_risk_detection_error(
        #     args.error_score, args.deadline, doc_trues, label_preds, early_stop_len_utt_list
        # )

        wandb.log(
            {
                "micro_F1": micro_F1,
                "Latency_F1": Latency_F1,
                # "ERDE": ERDE,
                "speed": metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p),
            }
        )

        return (
            classification_performance,
            micro_F1,
            risk_micro_F1,
            Latency_F1,
            # ERDE,
            early_stop_len_utt_list,
            doc_trues,
            label_preds,
        )

    def get_risk_num(self, queue):
        num_risk = len([1 for label in queue if label == "risk"])
        return num_risk


class SegBERTTrainer_chat_ann(Trainer):
    @torch.no_grad()
    def skeptical_predict(self, data_args, test_dataset: Dataset, device="cuda"):
        window_size = 50
        skeptical = 10
        pred_label_list = []
        early_stop_len_utt_list = []
        risk_fail = []
        normal_fail = 0
        sents_len = []  # 각 대화를 구성하는 전체 문장 수

        doc_labels = test_dataset["label"]

        self.tokenizer.truncation_side = "left"
        self.model.eval()

        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time
        wandb.init(
            project=data_args.task_name,
            entity="jinmyeong",
            name=name,
        )

        # 1. iterate test datset
        false_neg_datasets = []
        for test_datapoint in tqdm(test_dataset):

            label = test_datapoint["label"]
            queue = []
            pred_riskness = "normal"
            early_stop_len_utt = 0

            if label == 1:  # risk 인 대화의 발화 문장 개수
                # label == 'risk' 가 되어야하는거 아님?
                sents_len.append(len(test_datapoint["text"]))

            win_test_datapoint = build_utt_win_dataset(test_datapoint)
            win_test_dataloader = DataLoader(
                dataset=win_test_datapoint, batch_size=self.args.per_device_eval_batch_size, shuffle=False
            )  # , collate_fn=self.collate_fn)
            prob_list = torch.Tensor([])

            for idx, batch in enumerate(win_test_dataloader):
                encoded_batch = self.tokenizer(
                    batch["text"], padding="max_length", truncation=True, return_tensors="pt"
                )
                encoded_batch.to(device=self.model.device)
                probs = self.model(
                    input_ids=encoded_batch["input_ids"],
                    token_type_ids=encoded_batch["token_type_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                )
                prob_list = torch.cat((prob_list, probs.logits.cpu()), 0)
            for idx, prob in enumerate(prob_list):
                pred_riskness = "normal"
                early_stop_len_utt += 1
                pred_label = "normal" if np.argmax(prob, axis=0) == 0 else "risk"

                if len(queue) < 10:
                    queue.append(pred_label)
                else:
                    queue.pop(0)
                    queue.append(pred_label)

                risk_num = self.get_risk_num(queue)
                if risk_num >= 5:  # 📌 skeptical
                    pred_riskness = "risk"
                    break
            # determine risk or not
            pred_datapoint_label = 1 if pred_riskness == "risk" else 0
            early_stop_len_utt_list.append(early_stop_len_utt)

            pred_label_list.append(pred_datapoint_label)

            ############## Error Analysis ##############
            # true risk and pred normal
            if label == 1 and pred_datapoint_label == 0:
                risk_fail.append(1)
                # check false negative error
                false_neg_datasets.append(test_datapoint)
            # true normal and pred risk
            elif label == 0 and pred_datapoint_label == 1:
                normal_fail += 1
            # elif label == 1 and pred_datapoint_label == 1:
            #     print()
            # test_datapoint, win_test_datapoint[idx]
            ############################################

        # TODO: Evaluate recall, precision, F1, and Latency F1
        doc_trues = torch.tensor(doc_labels)  # data 개수 x 1
        label_preds = torch.tensor(pred_label_list)
        print("Detect 하지 못한 risk : \n", risk_fail)
        print("\nNum(risk_fail) : ", len(risk_fail))
        print("Num(normal_fail) : ", normal_fail)

        ############## Performace #################
        # classification_performace = metrics.classification_report(doc_trues, label_preds, target_names=['risk', 'normal'])
        classification_performance = metrics.classification_report(doc_trues, label_preds)
        micro_F1 = metrics.f1_score(doc_trues, label_preds, average="micro")
        risk_micro_F1 = metrics.f1_score(doc_trues, label_preds, pos_label=1, average="binary")

        p = np.log(3) * (1 / (np.median(sents_len) - 1))
        MESSAGE_WITH_HALF_PENALTY = 90
        # p = np.log(3)/(MESSAGE_WITH_HALF_PENALTY-1)

        Latency_F1 = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p) * metrics.f1_score(
            doc_trues, label_preds, pos_label=1, average="binary"
        )
        # metrics.f1_score(doc_trues, label_preds, pos_label=0, average='binary')
        # ERDE = metrics_test.early_risk_detection_error(
        #     args.error_score, args.deadline, doc_trues, label_preds, early_stop_len_utt_list
        # )

        wandb.log(
            {
                "micro_F1": micro_F1,
                "Latency_F1": Latency_F1,
                # "ERDE": ERDE,
                "speed": metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p),
            }
        )

        return (
            classification_performance,
            micro_F1,
            risk_micro_F1,
            Latency_F1,
            # ERDE,
            early_stop_len_utt_list,
            doc_trues,
            label_preds,
        )

    @torch.no_grad()
    def predict_chats(self):
        dataloader = self.get_eval_dataloader(eval_dataset=None)
        for idx, batch in tqdm(enumerate(dataloader)):
            batch = {k: v.to(device=self.model.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            outputs_prob = torch.nn.functional.softmax(outputs.logits, dim=1)

        return outputs_prob

    def get_risk_num(self, queue):
        num_risk = len([1 for label in queue if label == "risk"])
        return num_risk


class SegBERTTrainer_stage_ann(Trainer):
    @torch.no_grad()
    def skeptical_predict(self, data_args, test_dataset: Dataset, saved_path, device="cuda", threshold=5, window_size=10):
        window_size = 10
        skeptical = 10
        pred_label_list = []
        early_stop_len_utt_list = []
        risk_fail = []
        normal_fail = 0
        sents_len = []  # 각 대화를 구성하는 전체 문장 수
        # if type(test_dataset['labels'][0]) == int:
        #     doc_labels = test_dataset['labels']
        # else:
        #     doc_labels = [1 if real_label[0][0] == 'normal' else 0 for real_label in test_dataset['labels']]
        doc_labels = []
        strategy_true_label_list = []
        chat_true_label_list = []
        p_list = []
        chat_len_list = []

        self.tokenizer.truncation_side = "left"
        self.model.eval()

        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time
        wandb.init(
            project=data_args.task_name,
            entity="jinmyeong",
            name=name,
        )
        bert_result_list = []
        gold_strategy_label_list = []
        gold_chat_label_list = []

        # 1. iterate test datset
        for test_datapoint in tqdm(test_dataset):
            # chat length
            chat_len_list.append(len(test_datapoint["text"]))

            # calculate p ====================================
            stages_list = test_datapoint['stage']
            binary_stages_list = [1 if "G" in stage or "A" in stage or "R" in stage or "C" in stage or "I" in stage else 0 for stage in stages_list]
            num_neg_strategy = len([i for i in binary_stages_list if i == 1])
            latency = 0
            min_num_neg_strategy = 20
            

            if num_neg_strategy // 2 < min_num_neg_strategy: # chat에서는 20개의 neg strategy 까지만 봐줌
                min_num_neg_strategy = num_neg_strategy // 2

            count = 0

            for idx, binary_stages in enumerate(binary_stages_list):
                if binary_stages == 1:
                    count += 1
                if min_num_neg_strategy <= count:
                    latency = idx + 1

            p = math.log(3) / (latency - 1)
            p_list.append(p)
            # ====================================   

            utt_list = test_datapoint["text"]
            label = 1 if test_datapoint["label"] == "risk" else 0
            queue = []
            pred_riskness = "normal"
            early_stop_len_utt = 0

            if label == 1:  # risk 인 대화의 발화 문장 개수
                # label == 'risk' 가 되어야하는거 아님?
                sents_len.append(len(utt_list))

            win_test_datapoint = build_utt_win_dataset(test_datapoint)
            win_test_dataloader = DataLoader(
                dataset=win_test_datapoint, batch_size=self.args.per_device_eval_batch_size, shuffle=False
            )  # , collate_fn=self.collate_fn)
            prob_list = torch.Tensor([])

            for idx, batch in enumerate(win_test_dataloader):
                encoded_batch = self.tokenizer(
                    batch["text"], padding="max_length", truncation=True, return_tensors="pt"
                )
                encoded_batch.to(device=self.model.device)
                probs = self.model(
                    input_ids=encoded_batch["input_ids"],
                    #token_type_ids=encoded_batch["token_type_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                )
                prob_list = torch.cat((prob_list, probs.logits.cpu()), 0)

            # append bert_pred_label
            bert_pred_label = []
            gold_strategy_label = []
            gold_chat_label = []

            for idx, prob in enumerate(prob_list):
                pred_riskness = "normal"
                early_stop_len_utt += 1
                pred_label = "normal" if np.argmax(prob, axis=0) == 0 else "risk"

                # chat_label
                chat_label = test_datapoint["label"]
                # strategy_label
                strategy_label = (1
                if (
                    "G" in test_datapoint["stage"][idx]
                    or "A" in test_datapoint["stage"][idx]
                    or "C" in test_datapoint["stage"][idx]
                    or "R" in test_datapoint["stage"][idx]
                    or "I" in test_datapoint["stage"][idx]
                ) else 0)

                # append bert_pred_label
                bert_pred_label.append(pred_label)
                gold_strategy_label.append(strategy_label)
                gold_chat_label.append(chat_label)

            bert_result_list.append(bert_pred_label)
            gold_strategy_label_list.append(gold_strategy_label)
            gold_chat_label_list.append(gold_chat_label)

            # for idx, prob in enumerate(prob_list):
            #     pred_riskness = "normal"
            #     early_stop_len_utt += 1
            #     pred_label = "normal" if np.argmax(prob, axis=0) == 0 else "risk"


            #     if len(queue) < window_size:
            #         queue.append(pred_label)
            #     else:
            #         queue.pop(0)
            #         queue.append(pred_label)

            #     risk_num = self.get_risk_num(queue)
            #     if risk_num >= threshold:  # 5
            #         pred_riskness = "risk"
            #         break


            # pred_datapoint_label = 1 if pred_riskness == "risk" else 0
            # true_label = (
            #     1
            #     if (
            #         "G" in test_datapoint["stage"][idx]
            #         or "A" in test_datapoint["stage"][idx]
            #         or "C" in test_datapoint["stage"][idx]
            #         or "R" in test_datapoint["stage"][idx]
            #         or "I" in test_datapoint["stage"][idx]
            #     )
            #     and test_datapoint["label"] == 1
            #     else 0
            # )  # stage에 속한 utt일 경우

            # strategy_true_label = (
            #     1
            #     if "G" in test_datapoint["stage"][idx]
            #     or "A" in test_datapoint["stage"][idx]
            #     or "C" in test_datapoint["stage"][idx]
            #     or "R" in test_datapoint["stage"][idx]
            #     or "I" in test_datapoint["stage"][idx]
            #     else 0
            # )

            # chat_true_label = test_datapoint["label"]

            # strategy_true_label_list.append(strategy_true_label)
            # chat_true_label_list.append(chat_true_label)

            # doc_labels.append(true_label)            

            # early_stop_len_utt_list.append(early_stop_len_utt)

            # pred_label_list.append(pred_datapoint_label)

            # ############## Error Analysis ##############

            # # true risk and pred normal
            # if label == 1 and pred_datapoint_label == 0:
            #     risk_fail.append(1)
            # # true normal and pred risk
            # elif label == 0 and pred_datapoint_label == 1:
            #     normal_fail += 1
            # # true risk and pred risk
            # elif label == 1 and pred_datapoint_label == 1:
            #     print()
            # ############################################

        # save bert_result
        bert_outputs = []
        for bert_result, gold_strategy_label, gold_chat_label in zip(bert_result_list, gold_strategy_label_list, gold_chat_label_list):
            bert_outputs.append({
                "pred_label": bert_result,
                "gold_strategy": gold_strategy_label,
                "gold_chat_label": gold_chat_label
            })

        with open(f"{saved_path}/bert_result_list.jsonl", 'w') as file:
            for entry in bert_outputs:
                json.dump(entry, file)  # Convert dict to JSON string
                file.write('\n') 

        # # TODO: Evaluate recall, precision, F1, and Latency F1
        # doc_trues = torch.tensor(doc_labels)  # data 개수 x 1
        # label_preds = torch.tensor(pred_label_list)
        # strategy_trues = torch.tensor(strategy_true_label_list)
        # chat_trues = torch.tensor(chat_true_label_list)
        # print("Detect 하지 못한 risk : \n", risk_fail)
        # print("\nNum(risk_fail) : ", len(risk_fail))
        # print("Num(normal_fail) : ", normal_fail)

        # ############## Performace #################
        # # classification_performace = metrics.classification_report(doc_trues, label_preds, target_names=['risk', 'normal'])
        # strategy_classification_performance = metrics.classification_report(strategy_trues, label_preds)
        # chat_pred_classification_performance = metrics.classification_report(chat_trues, label_preds)

        # classification_performance = metrics.classification_report(doc_trues, label_preds)

        # micro_F1 = metrics.f1_score(doc_trues, label_preds, average="micro")
        # risk_micro_F1 = metrics.f1_score(doc_trues, label_preds, pos_label=1, average="binary")

        # # p = np.log(3) * (1 / (np.median(sents_len) - 1))
        # MESSAGE_WITH_HALF_PENALTY = 90
        # # p = np.log(3)/(MESSAGE_WITH_HALF_PENALTY-1)
        
        # Latency_F1 = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p_list) * metrics.f1_score(
        #     doc_trues, label_preds, pos_label=1, average="binary"
        # )
        # speed = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p_list)

        # # metrics.f1_score(doc_trues, label_preds, pos_label=0, average='binary')
        # # ERDE = metrics_test.early_risk_detection_error(
        # #     args.error_score, args.deadline, doc_trues, label_preds, early_stop_len_utt_list
        # # )

        # wandb.log(
        #     {
        #         "micro_F1": micro_F1,
        #         "Latency_F1": Latency_F1,
        #         # "ERDE": ERDE,
        #         "speed": metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p),
        #     }
        # )

        # print(f"classification_performance: {classification_performance}")
        # print(f"strategy_classification_performance: {strategy_classification_performance}")
        # print(f"chat_pred_classification_performance: {chat_pred_classification_performance}")
        # return (
        #     classification_performance,
        #     chat_pred_classification_performance,
        #     micro_F1,
        #     risk_micro_F1,
        #     Latency_F1,
        #     # ERDE,
        #     early_stop_len_utt_list,
        #     doc_trues,
        #     label_preds,
        #     speed
        # )

    def get_risk_num(self, queue):
        num_risk = len([1 for label in queue if label == "risk"])
        return num_risk


class ChatBERTTrainer_stage_ann(Trainer):
    @torch.no_grad()
    def skeptical_predict(self, data_args, test_dataset: Dataset, saved_path, device="cuda", threshold=5, window_size=10):
        # window_size = 10
        skeptical = 10
        pred_label_list = []
        early_stop_len_utt_list = []
        risk_fail = []
        normal_fail = 0
        sents_len = []  # 각 대화를 구성하는 전체 문장 수
        doc_labels = []
        strategy_true_label_list = []
        chat_true_label_list = []
        p_list = []
        chat_len_list = []

        self.tokenizer.truncation_side = "left"
        self.model.eval()

        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time
        wandb.init(
            project=data_args.task_name,
            entity="jinmyeong",
            name=name,
        )
        bert_result_list = []
        gold_strategy_label_list = []
        gold_chat_label_list = []

        # 1. iterate test datset
        for test_datapoint in tqdm(test_dataset):
            # chat length
            chat_len_list.append(len(test_datapoint["text"]))

            # calculate p ====================================
            stages_list = test_datapoint['stage']
            binary_stages_list = [1 if "G" in stage or "A" in stage or "R" in stage or "C" in stage or "I" in stage else 0 for stage in stages_list]
            num_neg_strategy = len([i for i in binary_stages_list if i == 1])
            latency = 0
            min_num_neg_strategy = 20
            

            if num_neg_strategy // 2 < min_num_neg_strategy: # chat에서는 20개의 neg strategy 까지만 봐줌
                min_num_neg_strategy = num_neg_strategy // 2

            count = 0

            for idx, binary_stages in enumerate(binary_stages_list):
                if binary_stages == 1:
                    count += 1
                if min_num_neg_strategy <= count:
                    latency = idx + 1

            p = math.log(3) / (latency - 1)
            p_list.append(p)
            # ====================================   
            utt_list = test_datapoint["text"]
            label = test_datapoint["label"]
            # label = 1 if test_datapoint['labels'][0][0] == 'risk' else 0
            queue = []
            pred_riskness = "normal"
            early_stop_len_utt = 0

            if label == 1:  # risk 인 대화의 발화 문장 개수
                # label == 'risk' 가 되어야하는거 아님?
                sents_len.append(len(utt_list))

            win_test_datapoint = build_utt_win_dataset_no_SEP(test_datapoint)
            win_test_dataloader = DataLoader(
                dataset=win_test_datapoint, batch_size=self.args.per_device_eval_batch_size, shuffle=False
            )  # , collate_fn=self.collate_fn)
            prob_list = torch.Tensor([])

            for idx, batch in enumerate(win_test_dataloader):
                encoded_batch = self.tokenizer(
                    batch["text"], padding="max_length", truncation=True, return_tensors="pt"
                )
                # encoded_batch = self.tokenizer(
                #     batch["text"], padding=True, truncation=True, return_tensors="pt"
                # )
                encoded_batch.to(device=self.model.device)
                probs = self.model(
                    input_ids=encoded_batch["input_ids"],
                    token_type_ids=encoded_batch["token_type_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                )
                prob_list = torch.cat((prob_list, probs.logits.cpu()), 0)
            
            # append bert_pred_label
            bert_pred_label = []
            gold_strategy_label = []
            gold_chat_label = []

            for idx, prob in enumerate(prob_list):
                pred_riskness = "normal"
                early_stop_len_utt += 1
                pred_label = "normal" if np.argmax(prob, axis=0) == 0 else "risk"

                # chat_label
                chat_label = test_datapoint["label"]
                # strategy_label
                strategy_label = (1
                if (
                    "G" in test_datapoint["stage"][idx]
                    or "A" in test_datapoint["stage"][idx]
                    or "C" in test_datapoint["stage"][idx]
                    or "R" in test_datapoint["stage"][idx]
                    or "I" in test_datapoint["stage"][idx]
                ) else 0)

                # append bert_pred_label
                bert_pred_label.append(pred_label)
                gold_strategy_label.append(strategy_label)
                gold_chat_label.append(chat_label)

            bert_result_list.append(bert_pred_label)
            gold_strategy_label_list.append(gold_strategy_label)
            gold_chat_label_list.append(gold_chat_label)

            for idx, prob in enumerate(prob_list):
                pred_riskness = "normal"
                early_stop_len_utt += 1
                pred_label = "normal" if np.argmax(prob, axis=0) == 0 else "risk"
                # append bert_pred_label

                if len(queue) < window_size:
                    queue.append(pred_label)
                else:
                    queue.pop(0)
                    queue.append(pred_label)

                risk_num = self.get_risk_num(queue)
                if risk_num >= threshold:  # 5
                    pred_riskness = "risk"
                    break

                    
            # determine risk or not
            pred_datapoint_label = 1 if pred_riskness == "risk" else 0
            true_label = (
                1
                if (
                    "G" in test_datapoint["stage"][idx]
                    or "A" in test_datapoint["stage"][idx]
                    or "C" in test_datapoint["stage"][idx]
                    or "R" in test_datapoint["stage"][idx]
                    or "I" in test_datapoint["stage"][idx]
                )
                and test_datapoint["label"] == 1
                else 0
            )  # stage에 속한 utt일 경우

            strategy_true_label = (
                1
                if "G" in test_datapoint["stage"][idx]
                or "A" in test_datapoint["stage"][idx]
                or "C" in test_datapoint["stage"][idx]
                or "R" in test_datapoint["stage"][idx]
                or "I" in test_datapoint["stage"][idx]
                else 0
            )

            chat_true_label = test_datapoint["label"]

            strategy_true_label_list.append(strategy_true_label)
            chat_true_label_list.append(chat_true_label)

            doc_labels.append(true_label)

            early_stop_len_utt_list.append(early_stop_len_utt)

            pred_label_list.append(pred_datapoint_label)

            ############## Error Analysis ##############
            # true risk and pred normal
            if label == 1 and pred_datapoint_label == 0:
                risk_fail.append(1)
            # true normal and pred risk
            elif label == 0 and pred_datapoint_label == 1:
                normal_fail += 1
            ############################################

        # save bert_result
        bert_outputs = []
        for bert_result, gold_strategy_label, gold_chat_label in zip(bert_result_list, gold_strategy_label_list, gold_chat_label_list):
            bert_outputs.append({
                "pred_label": bert_result,
                "gold_strategy": gold_strategy_label,
                "gold_chat_label": gold_chat_label
            })

        with open(f"{saved_path}/bert_result_list.jsonl", 'w') as file:
            for entry in bert_outputs:
                json.dump(entry, file)  # Convert dict to JSON string
                file.write('\n')          

        # TODO: Evaluate recall, precision, F1, and Latency F1
        doc_trues = torch.tensor(doc_labels)  # data 개수 x 1
        label_preds = torch.tensor(pred_label_list)
        strategy_trues = torch.tensor(strategy_true_label_list)
        chat_trues = torch.tensor(chat_true_label_list)

        print("Detect 하지 못한 risk : \n", risk_fail)
        print("\nNum(risk_fail) : ", len(risk_fail))
        print("Num(normal_fail) : ", normal_fail)

        ############## Performace #################
        # classification_performace = metrics.classification_report(doc_trues, label_preds, target_names=['risk', 'normal'])
        strategy_classification_performance = metrics.classification_report(strategy_trues, label_preds)
        chat_pred_classification_performance = metrics.classification_report(chat_trues, label_preds)

        classification_performance = metrics.classification_report(doc_trues, label_preds)
        micro_F1 = metrics.f1_score(doc_trues, label_preds, average="micro")
        risk_micro_F1 = metrics.f1_score(doc_trues, label_preds, pos_label=1, average="binary")

        # p = np.log(3) * (1 / (np.median(sents_len) - 1))
        MESSAGE_WITH_HALF_PENALTY = 90
        # p = np.log(3)/(MESSAGE_WITH_HALF_PENALTY-1)

        Latency_F1 = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p_list) * metrics.f1_score(
            doc_trues, label_preds, pos_label=1, average="binary"
        )
        speed = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p_list)
        # metrics.f1_score(doc_trues, label_preds, pos_label=0, average='binary')
        # ERDE = self.early_risk_detection_error(
        #     error_score=[1,1,1,1], deadline=40, doc_trues=chat_trues, label_preds=label_preds, early_stop_len_utt_list=early_stop_len_utt_list
        # )
        # early_rate = self.get_early_rate(chat_len_list=chat_len_list, early_stop_len_utt_list=early_stop_len_utt_list)

        def compute_error(label_preds, chat_trues, strategy_trues, early_stop_len_utt_list):
            result = []
            FN, FP, TN, TP = 0, 0, 0, 0

            for idx in range(len(label_preds)):
                early_idx = early_stop_len_utt_list[idx] - 1
                utt_chunk = test_dataset[idx]["text"][: early_idx + 1]
                utt_chunk = utt_chunk[-30:]

                real_label = int(chat_trues[idx])
                predict_label = int(label_preds[idx])
                stage_label = int(strategy_trues[idx])

                if real_label == 1 and predict_label == 1:
                    TP += 1
                if real_label == 1 and predict_label == 0:
                    FN += 1
                if real_label == 0 and predict_label == 0:
                    TN += 1
                if real_label == 0 and predict_label == 1:
                    FP += 1

                result.append(
                    {
                        "id": f"index={idx}",
                        "early_idx": early_idx,
                        "utt_chunk": utt_chunk,
                        "real_label": real_label,
                        "predict_label": predict_label,
                        "stage_label": stage_label,
                    }
                )
            print(f"False negative Error rate: {FN/(FN+TP)}")
            print(f"False positive Error rate: {FP/(FP+TN)}")
            print(f"True negative Error rate: {TN/(FP+TN)}")
            print(f"True positive Error rate: {TP/(FN+TP)}")

            return result

        # result = compute_error(label_preds, chat_trues, strategy_trues, early_stop_len_utt_list)
        # chatBERT_result = {"chatBERT_result": result}
        # with open("/home/jinmyeong/code/eSPD/data/chatBERT_result_w_stage.json", "w") as f:
        #     json.dump(chatBERT_result, f, indent=2)

        # wandb.log(
        #     {
        #         "micro_F1": micro_F1,
        #         "Latency_F1": Latency_F1,
        #         # "ERDE": ERDE,
        #         "speed": metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p),
        #     }
        # )

        print(f"classification_performance: {classification_performance}")
        print(f"strategy_classification_performance: {strategy_classification_performance}")
        print(f"chat_pred_classification_performance: {chat_pred_classification_performance}")
        self.save_test_output(doc_trues.tolist(), label_preds.tolist(), [d-1 for d in early_stop_len_utt_list], 
                                 p_list, strategy_trues.tolist(), chat_trues.tolist(), 
                                 saved_path, classification_performance, chat_pred_classification_performance,
                                 speed, Latency_F1)
        return (
            classification_performance,
            chat_pred_classification_performance,
            micro_F1,
            risk_micro_F1,
            Latency_F1,
            # ERDE,
            early_stop_len_utt_list,
            doc_trues,
            label_preds,
            speed
        )
    def save_test_output(self, doc_trues, early_preds, delay_list, 
                            p_list, strategy_trues, chat_trues, 
                            saved_path, classification_performance, chat_pred_classification_performance,
                            speed, Latency_F1):
        outputs = []
        for doc_true, early_pred, delay_idx, p, strategy_true, chat_true in zip(doc_trues, early_preds, delay_list, p_list, strategy_trues, chat_trues):
            output = {"doc_true": doc_true, 
                    "early_pred": early_pred, 
                    "delay_idx": delay_idx, 
                    "p": p, 
                    "strategy_true": strategy_true, 
                    "chat_true": chat_true}
            outputs.append(output)

        with open(f"{saved_path}/outputs.jsonl", 'w') as file:
            for entry in outputs:
                json.dump(entry, file)  # Convert dict to JSON string
                file.write('\n')  
        

        results = [{"classification_performance": classification_performance,
                "chat_pred_classification_performance": chat_pred_classification_performance,
                "speed": speed,
                "Latency_F1": Latency_F1}]
        
        with open(f"{saved_path}/results.jsonl", 'w') as file:
            for entry in results:
                json.dump(entry, file)  # Convert dict to JSON string
                file.write('\n')  

        return outputs
    def early_risk_detection_error(self, error_score, deadline, doc_trues, label_preds, delay_list):
        # error_score = args.error_score  : [fp, fn, tp]
        error_score = [1, 1, 1]
        deadline = 40

        erde = 0
        for gold, pred, delay in zip(doc_trues, label_preds, delay_list):
            # true positive => multiply latency cost function
            if gold == 1 and pred == 1:
                erde += self.lc(deadline, delay) * error_score[2]
            # false negative
            elif gold == 1 and pred == 0:
                erde += error_score[1]
            # false positive
            elif gold == 0 and pred == 1:
                erde += error_score[0]
            # true negative 인 경우 erde = 0
        erde = erde / len(doc_trues)

        return erde
    
    def lc(self, deadline, delay):
        return 1 - (1 / (1 + np.exp(delay - deadline)))
    
    def get_early_rate(self, chat_len_list, delay_list): # 수정 필요 -> pred_label = 1 and label = 1
        # chat_len_list = [199, 1000, ...]
        # delay_list = [5, 14, ...]
        early_rate_list = []

        for chat_len, delay in zip(chat_len_list, delay_list):
            er = chat_len / delay
            early_rate_list.append(er)
        
        early_rate = sum(early_rate_list) / len(chat_len_list)
        return early_rate
    
    def get_risk_num(self, queue):
        num_risk = len([1 for label in queue if label == "risk"])
        return num_risk


class SegBERTTrainer_stage_ann_speaker(Trainer):
    @torch.no_grad()
    def skeptical_predict(self, data_args, test_dataset: Dataset, device="cuda"):
        window_size = 50
        skeptical = 10
        pred_label_list = []
        early_stop_len_utt_list = []
        risk_fail = []
        normal_fail = 0
        sents_len = []  # 각 대화를 구성하는 전체 문장 수
        # if type(test_dataset['labels'][0]) == int:
        #     doc_labels = test_dataset['labels']
        # else:
        #     doc_labels = [1 if real_label[0][0] == 'normal' else 0 for real_label in test_dataset['labels']]
        doc_labels = []

        self.tokenizer.truncation_side = "left"
        self.model.eval()

        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time
        wandb.init(
            project=data_args.task_name,
            entity="jinmyeong",
            name=name,
        )
        # 1. iterate test datset
        for test_datapoint in tqdm(test_dataset):
            utt_list = test_datapoint["text"]
            # label = test_datapoint["label"]
            label = 1 if test_datapoint["label"] == "risk" else 0
            # label = 1 if test_datapoint['labels'][0][0] == 'risk' else 0
            queue = []
            pred_riskness = "normal"
            early_stop_len_utt = 0

            if label == 1:  # risk 인 대화의 발화 문장 개수
                # label == 'risk' 가 되어야하는거 아님?
                sents_len.append(len(utt_list))

            win_test_datapoint = self.build_utt_win_dataset(test_datapoint)
            win_test_dataloader = DataLoader(
                dataset=win_test_datapoint, batch_size=self.args.per_device_eval_batch_size, shuffle=False
            )  # , collate_fn=self.collate_fn)
            prob_list = torch.Tensor([])

            for idx, batch in enumerate(win_test_dataloader):
                encoded_batch = self.tokenizer(
                    batch["text"], padding="max_length", truncation=True, return_tensors="pt"
                )
                encoded_batch.to(device=self.model.device)
                probs = self.model(
                    input_ids=encoded_batch["input_ids"],
                    token_type_ids=encoded_batch["token_type_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                )
                prob_list = torch.cat((prob_list, probs.logits.cpu()), 0)
            for idx, prob in enumerate(prob_list):
                pred_riskness = "normal"
                early_stop_len_utt += 1
                pred_label = "normal" if np.argmax(prob, axis=0) == 0 else "risk"

                # for idx, utt in enumerate(utt_list):
                #     pred_riskness = "normal"
                #     early_stop_len_utt += 1

                #     window = utt_list[max(0, idx + 1 - window_size) : idx + 1]
                #     text = " ".join(window)
                #     encoded_text = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

                #     input_ids = encoded_text["input_ids"].to(device)
                #     token_type_ids = encoded_text["token_type_ids"].to(device)
                #     attention_mask = encoded_text["attention_mask"].to(device)

                #     prob = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                #     pred_label = "normal" if np.argmax(prob.logits.cpu(), axis=1) == 0 else "risk"
                #     # pred_label = 'normal' if np.argmax(prob.logits.cpu(), axis=1) == 0 else 'risk'#

                if len(queue) < 10:
                    queue.append(pred_label)
                else:
                    queue.pop(0)
                    queue.append(pred_label)

                risk_num = self.get_risk_num(queue)
                if risk_num >= 5:  # 5
                    pred_riskness = "risk"
                    break

            # determine risk or not
            pred_datapoint_label = 1 if pred_riskness == "risk" else 0
            true_label = 1 if "G" in test_datapoint["stage"][idx] else 0  # stage에 속한 utt일 경우
            # true_label = 1 if len(test_datapoint["stage"][idx]) >= 1 else 0
            doc_labels.append(true_label)

            early_stop_len_utt_list.append(early_stop_len_utt)

            pred_label_list.append(pred_datapoint_label)

            ############## Error Analysis ##############

            # true risk and pred normal
            if label == 1 and pred_datapoint_label == 0:
                risk_fail.append(1)
            # true normal and pred risk
            elif label == 0 and pred_datapoint_label == 1:
                normal_fail += 1
            # true risk and pred risk
            elif label == 1 and pred_datapoint_label == 1:
                print()
            ############################################

        # TODO: Evaluate recall, precision, F1, and Latency F1
        doc_trues = torch.tensor(doc_labels)  # data 개수 x 1
        label_preds = torch.tensor(pred_label_list)
        print("Detect 하지 못한 risk : \n", risk_fail)
        print("\nNum(risk_fail) : ", len(risk_fail))
        print("Num(normal_fail) : ", normal_fail)

        ############## Performace #################
        # classification_performace = metrics.classification_report(doc_trues, label_preds, target_names=['risk', 'normal'])
        classification_performance = metrics.classification_report(doc_trues, label_preds)
        micro_F1 = metrics.f1_score(doc_trues, label_preds, average="micro")
        risk_micro_F1 = metrics.f1_score(doc_trues, label_preds, pos_label=1, average="binary")

        p = np.log(3) * (1 / (np.median(sents_len) - 1))
        MESSAGE_WITH_HALF_PENALTY = 90
        # p = np.log(3)/(MESSAGE_WITH_HALF_PENALTY-1)

        Latency_F1 = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p) * metrics.f1_score(
            doc_trues, label_preds, pos_label=1, average="binary"
        )
        # metrics.f1_score(doc_trues, label_preds, pos_label=0, average='binary')
        # ERDE = metrics_test.early_risk_detection_error(
        #     args.error_score, args.deadline, doc_trues, label_preds, early_stop_len_utt_list
        # )

        wandb.log(
            {
                "micro_F1": micro_F1,
                "Latency_F1": Latency_F1,
                # "ERDE": ERDE,
                "speed": metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p),
            }
        )

        return (
            classification_performance,
            micro_F1,
            risk_micro_F1,
            Latency_F1,
            # ERDE,
            early_stop_len_utt_list,
            doc_trues,
            label_preds,
        )

    def get_risk_num(self, queue):
        num_risk = len([1 for label in queue if label == "risk"])
        return num_risk

    def build_utt_win_dataset(self, raw_dataset, window_size=50):
        win_raw_dataset = []

        utterances = raw_dataset["text"]
        authors = raw_dataset["author"]

        speaker_utterances = []
        for utterance, speaker in zip(utterances, authors):
            # Insert the speaker token before and after each utterance
            speaker_utterance = f"[{speaker}]{utterance}"
            speaker_utterances.append(speaker_utterance)

        # label = raw_dataset["labels"][0][0]
        start_idx = 0
        end_idx = 0
        counter = -1
        # 시작 길이가 50으로 고정인 경우:
        # for idx, utterance in enumerate(utterances):
        #     if window_size >= len(utterances) - idx:
        #         win_utterences = utterances[idx:]
        #         win_raw_dataset = get_win_raw_dataset(win_utterences, label)
        #         win_raw_dataset.append(win_raw_dataset)
        #         break

        #     win_utterences = utterances[idx:window_size + idx]
        #     win_raw_dataset = get_win_raw_dataset(win_utterences, label)
        #     win_raw_dataset.append(win_raw_dataset)

        # 시작 길이가 1인 경우:
        while True:
            counter += 1
            if end_idx >= len(speaker_utterances):
                break
            if start_idx < end_idx:
                win_utterences = " ".join(speaker_utterances[start_idx:end_idx])
                win_utterences += " [SEP] "
                win_utterences += speaker_utterances[end_idx]
            else:
                win_utterences = " ".join(speaker_utterances[start_idx : end_idx + 1])
            # win_utterences = ' '.join(utterances[start_idx:end_idx+1])
            # win_raw_dataset = get_win_raw_dataset(win_utterences=win_utterences, label='risk' if raw_dataset['utt_annotation'][counter] == 1 else 'normal')

            # label = 1 if len(raw_dataset['stage'][counter]) >= 1 else 0 # 📌 어떤 stage를 위험하다고 할지 -> PI, G, A
            label = 1 if "G" in raw_dataset["stage"][counter] else 0  # 📌 어떤 stage를 위험하다고 할지 -> G, A
            win_datapoint = self.get_win_datapoint(win_utterences=win_utterences, label=label)
            win_raw_dataset.append(win_datapoint)

            end_idx += 1
            if window_size < end_idx - start_idx + 1:
                start_idx += 1

        return win_raw_dataset

    def get_win_datapoint(self, win_utterences, label=None):
        if label == 1 or label == 0:
            win_datapoint = {"text": win_utterences, "label": label}
        else:
            win_datapoint = {"text": win_utterences}

        return win_datapoint


class SegBERTTrainer_allStage_ann_speaker(Trainer):
    @torch.no_grad()
    def skeptical_predict(self, data_args, test_dataset: Dataset, device="cuda"):
        window_size = 50
        skeptical = 10
        pred_label_list = []
        early_stop_len_utt_list = []
        risk_fail = []
        normal_fail = 0
        sents_len = []  # 각 대화를 구성하는 전체 문장 수
        # if type(test_dataset['labels'][0]) == int:
        #     doc_labels = test_dataset['labels']
        # else:
        #     doc_labels = [1 if real_label[0][0] == 'normal' else 0 for real_label in test_dataset['labels']]
        doc_labels = []

        self.tokenizer.truncation_side = "left"
        self.model.eval()

        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time
        wandb.init(
            project=data_args.task_name,
            entity="jinmyeong",
            name=name,
        )
        # 1. iterate test datset
        for test_datapoint in tqdm(test_dataset):
            utt_list = test_datapoint["text"]
            # label = test_datapoint["label"]
            label = 1 if test_datapoint["label"] == "risk" else 0
            # label = 1 if test_datapoint['labels'][0][0] == 'risk' else 0
            queue = []
            pred_riskness = "normal"
            early_stop_len_utt = 0

            if label == 1:  # risk 인 대화의 발화 문장 개수
                # label == 'risk' 가 되어야하는거 아님?
                sents_len.append(len(utt_list))

            win_test_datapoint = self.build_utt_win_dataset(test_datapoint)
            win_test_dataloader = DataLoader(
                dataset=win_test_datapoint, batch_size=self.args.per_device_eval_batch_size, shuffle=False
            )  # , collate_fn=self.collate_fn)
            prob_list = torch.Tensor([])

            for idx, batch in enumerate(win_test_dataloader):
                encoded_batch = self.tokenizer(
                    batch["text"], padding="max_length", truncation=True, return_tensors="pt"
                )
                encoded_batch.to(device=self.model.device)
                probs = self.model(
                    input_ids=encoded_batch["input_ids"],
                    token_type_ids=encoded_batch["token_type_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                )
                prob_list = torch.cat((prob_list, probs.logits.cpu()), 0)
            for idx, prob in enumerate(prob_list):
                pred_riskness = "normal"
                early_stop_len_utt += 1
                pred_label = "risk" if np.argmax(prob, axis=0) == 1 else "normal"

                # for idx, utt in enumerate(utt_list):
                #     pred_riskness = "normal"
                #     early_stop_len_utt += 1

                #     window = utt_list[max(0, idx + 1 - window_size) : idx + 1]
                #     text = " ".join(window)
                #     encoded_text = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

                #     input_ids = encoded_text["input_ids"].to(device)
                #     token_type_ids = encoded_text["token_type_ids"].to(device)
                #     attention_mask = encoded_text["attention_mask"].to(device)

                #     prob = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                #     pred_label = "normal" if np.argmax(prob.logits.cpu(), axis=1) == 0 else "risk"
                #     # pred_label = 'normal' if np.argmax(prob.logits.cpu(), axis=1) == 0 else 'risk'#

                if len(queue) < 10:
                    queue.append(pred_label)
                else:
                    queue.pop(0)
                    queue.append(pred_label)

                risk_num = self.get_risk_num(queue)
                if risk_num >= 7:  # 5
                    pred_riskness = "risk"
                    break

            # determine risk or not
            pred_datapoint_label = 1 if pred_riskness == "risk" else 0
            true_label = 1 if "G" in test_datapoint["stage"][idx] else 0  # stage에 속한 utt일 경우
            # true_label = 1 if len(test_datapoint["stage"][idx]) >= 1 else 0
            doc_labels.append(true_label)

            early_stop_len_utt_list.append(early_stop_len_utt)

            pred_label_list.append(pred_datapoint_label)

            ############## Error Analysis ##############

            # true risk and pred normal
            if label == 1 and pred_datapoint_label == 0:
                risk_fail.append(1)
            # true normal and pred risk
            elif label == 0 and pred_datapoint_label == 1:
                normal_fail += 1
            # true risk and pred risk
            elif label == 1 and pred_datapoint_label == 1:
                print()
            ############################################

        # TODO: Evaluate recall, precision, F1, and Latency F1
        doc_trues = torch.tensor(doc_labels)  # data 개수 x 1
        label_preds = torch.tensor(pred_label_list)
        print("Detect 하지 못한 risk : \n", risk_fail)
        print("\nNum(risk_fail) : ", len(risk_fail))
        print("Num(normal_fail) : ", normal_fail)

        ############## Performace #################
        # classification_performace = metrics.classification_report(doc_trues, label_preds, target_names=['risk', 'normal'])
        classification_performance = metrics.classification_report(doc_trues, label_preds)
        micro_F1 = metrics.f1_score(doc_trues, label_preds, average="micro")
        risk_micro_F1 = metrics.f1_score(doc_trues, label_preds, pos_label=1, average="binary")

        p = np.log(3) * (1 / (np.median(sents_len) - 1))
        MESSAGE_WITH_HALF_PENALTY = 90
        # p = np.log(3)/(MESSAGE_WITH_HALF_PENALTY-1)

        Latency_F1 = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p) * metrics.f1_score(
            doc_trues, label_preds, pos_label=1, average="binary"
        )
        # metrics.f1_score(doc_trues, label_preds, pos_label=0, average='binary')
        # ERDE = metrics_test.early_risk_detection_error(
        #     args.error_score, args.deadline, doc_trues, label_preds, early_stop_len_utt_list
        # )

        wandb.log(
            {
                "micro_F1": micro_F1,
                "Latency_F1": Latency_F1,
                # "ERDE": ERDE,
                "speed": metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p),
            }
        )

        return (
            classification_performance,
            micro_F1,
            risk_micro_F1,
            Latency_F1,
            # ERDE,
            early_stop_len_utt_list,
            doc_trues,
            label_preds,
        )

    def get_risk_num(self, queue):
        num_risk = len([1 for label in queue if label == "risk"])
        return num_risk

    def build_utt_win_dataset(self, raw_dataset, window_size=50):
        win_raw_dataset = []

        utterances = raw_dataset["text"]
        authors = raw_dataset["author"]

        speaker_utterances = []
        for utterance, speaker in zip(utterances, authors):
            # Insert the speaker token before and after each utterance
            speaker_utterance = f"[{speaker}]{utterance}"
            speaker_utterances.append(speaker_utterance)

        # label = raw_dataset["labels"][0][0]
        start_idx = 0
        end_idx = 0
        counter = -1
        # 시작 길이가 50으로 고정인 경우:
        # for idx, utterance in enumerate(utterances):
        #     if window_size >= len(utterances) - idx:
        #         win_utterences = utterances[idx:]
        #         win_raw_dataset = get_win_raw_dataset(win_utterences, label)
        #         win_raw_dataset.append(win_raw_dataset)
        #         break

        #     win_utterences = utterances[idx:window_size + idx]
        #     win_raw_dataset = get_win_raw_dataset(win_utterences, label)
        #     win_raw_dataset.append(win_raw_dataset)

        # 시작 길이가 1인 경우:
        while True:
            counter += 1
            if end_idx >= len(speaker_utterances):
                break
            if start_idx < end_idx:
                win_utterences = " ".join(speaker_utterances[start_idx:end_idx])
                win_utterences += " [SEP] "
                win_utterences += speaker_utterances[end_idx]
            else:
                win_utterences = " ".join(speaker_utterances[start_idx : end_idx + 1])
            # win_utterences = ' '.join(utterances[start_idx:end_idx+1])
            # win_raw_dataset = get_win_raw_dataset(win_utterences=win_utterences, label='risk' if raw_dataset['utt_annotation'][counter] == 1 else 'normal')

            # label = 1 if len(raw_dataset['stage'][counter]) >= 1 else 0 # 📌 어떤 stage를 위험하다고 할지 -> PI, G, A
            label = 1 if "G" in raw_dataset["stage"][counter] else 0  # 📌 어떤 stage를 위험하다고 할지 -> G, A
            win_datapoint = self.get_win_datapoint(win_utterences=win_utterences, label=label)
            win_raw_dataset.append(win_datapoint)

            end_idx += 1
            if window_size < end_idx - start_idx + 1:
                start_idx += 1

        return win_raw_dataset

    def get_win_datapoint(self, win_utterences, label=None):
        if label == 1 or label == 0:
            win_datapoint = {"text": win_utterences, "label": label}
        else:
            win_datapoint = {"text": win_utterences}

        return win_datapoint


class SegBERTTrainer_stage_ann_chat_seg(Trainer):
    @torch.no_grad()
    def skeptical_predict(self, data_args, test_dataset: Dataset, device="cuda", threshold=5, window_size=10):
        window_size = 50
        skeptical = 8
        pred_label_list = []
        early_stop_len_utt_list = []
        strategy_true_label_list = []
        chat_true_label_list = []
        risk_fail = []
        normal_fail = 0
        sents_len = []  # 각 대화를 구성하는 전체 문장 수
        p_list = []
        chat_len_list = []

        # if type(test_dataset['labels'][0]) == int:
        #     doc_labels = test_dataset['labels']
        # else:
        #     doc_labels = [1 if real_label[0][0] == 'normal' else 0 for real_label in test_dataset['labels']]
        doc_labels = []

        self.tokenizer.truncation_side = "left"
        self.model.eval()

        kor_time = (datetime.now() + timedelta(hours=9)).strftime("%m%d%H%M")
        name = kor_time
        wandb.init(
            project=data_args.task_name,
            entity="jinmyeong",
            name=name,
        )
        # 1. iterate test datset
        for test_datapoint in tqdm(test_dataset):
            # chat length
            chat_len_list.append(len(test_datapoint["text"]))

            # calculate p ====================================
            stages_list = test_datapoint['stage']
            binary_stages_list = [1 if "G" in stage or "A" in stage or "R" in stage or "C" in stage or "I" in stage else 0 for stage in stages_list]
            num_neg_strategy = len([i for i in binary_stages_list if i == 1])
            latency = 0
            min_num_neg_strategy = 20
            

            if num_neg_strategy // 2 < min_num_neg_strategy: # chat에서는 20개의 neg strategy 까지만 봐줌
                min_num_neg_strategy = num_neg_strategy // 2

            count = 0

            for idx, binary_stages in enumerate(binary_stages_list):
                if binary_stages == 1:
                    count += 1
                if min_num_neg_strategy <= count:
                    latency = idx + 1

            p = math.log(3) / (latency - 1)
            p_list.append(p)
            # ====================================   
            utt_list = test_datapoint["text"]
            label = test_datapoint["label"]
            # label = 1 if test_datapoint["label"] == "risk" else 0
            # label = 1 if test_datapoint['labels'][0][0] == 'risk' else 0
            queue = []
            pred_riskness = "normal"
            early_stop_len_utt = 0

            if label == 1:  # risk 인 대화의 발화 문장 개수
                # label == 'risk' 가 되어야하는거 아님?
                sents_len.append(len(utt_list))

            win_test_datapoint = build_utt_win_dataset(test_datapoint)
            win_test_dataloader = DataLoader(
                dataset=win_test_datapoint, batch_size=self.args.per_device_eval_batch_size, shuffle=False
            )  # , collate_fn=self.collate_fn)
            prob_list = torch.Tensor([])

            for idx, batch in enumerate(win_test_dataloader):
                encoded_batch = self.tokenizer(
                    batch["text"], padding="max_length", truncation=True, return_tensors="pt"
                )
                encoded_batch.to(device=self.model.device)
                probs = self.model(
                    input_ids=encoded_batch["input_ids"],
                    token_type_ids=encoded_batch["token_type_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                )  # torch.Size([batch_size, 4])
                prob_list = torch.cat((prob_list, probs.logits.cpu()), 0)
            for idx, prob in enumerate(prob_list):
                pred_riskness = "normal"
                early_stop_len_utt += 1
                # {"PI": 0, "G": 1, "A": 2, "O": 3, "C": 1, "R": 1, "I": 2}
                if np.argmax(prob, axis=0) == 1 or np.argmax(prob, axis=0) == 2:
                    pred_label = "risk"
                else:
                    pred_label = "normal"

                if len(queue) < window_size:
                    queue.append(pred_label)
                else:
                    queue.pop(0)
                    queue.append(pred_label)

                risk_num = self.get_risk_num(queue)
                if risk_num >= threshold:  # 5
                    pred_riskness = "risk"
                    break

            # determine risk or not
            pred_datapoint_label = 1 if pred_riskness == "risk" else 0
            true_label = (
                1
                if (
                    "G" in test_datapoint["stage"][idx]
                    or "C" in test_datapoint["stage"][idx]
                    or "R" in test_datapoint["stage"][idx]
                    or "A" in test_datapoint["stage"][idx]
                    or "I" in test_datapoint["stage"][idx]
                )
                and label == 1
                else 0
            )  # stage에 속한 utt일 경우

            # if pred_datapoint_label == 0 and label == 1:  # 📌 다 봤는데 마지막 utterance가 normal인데 전체가 risk일 수 있으니!
            #     true_label = 1  # TODO: 이상한데? -> 현재 idx가 마지막인 조건도 넣어야지

            if (
                len(utt_list) == idx + 1 and pred_datapoint_label == 0 and label == 1
            ):  # 📌 다 봤는데 마지막 utterance가 normal인데 전체가 risk일 수 있으니!
                true_label = 1

            strategy_true_label = (
                1
                if (
                    "G" in test_datapoint["stage"][idx]
                    or "C" in test_datapoint["stage"][idx]
                    or "A" in test_datapoint["stage"][idx]
                    or "I" in test_datapoint["stage"][idx]
                    or "R" in test_datapoint["stage"][idx]
                )
                else 0
            )
            chat_true_label = label

            doc_labels.append(true_label)
            early_stop_len_utt_list.append(early_stop_len_utt)
            pred_label_list.append(pred_datapoint_label)

            strategy_true_label_list.append(strategy_true_label)
            chat_true_label_list.append(chat_true_label)
            ############## Error Analysis ##############

            # true risk and pred normal
            if label == 1 and pred_datapoint_label == 0:
                risk_fail.append(1)
            # true normal and pred risk
            elif label == 0 and pred_datapoint_label == 1:
                normal_fail += 1
            # true risk and pred risk
            elif label == 1 and pred_datapoint_label == 1:
                print()
            ############################################

        # TODO: Evaluate recall, precision, F1, and Latency F1
        doc_trues = torch.tensor(doc_labels)  # data 개수 x 1
        label_preds = torch.tensor(pred_label_list)
        strategy_trues = torch.tensor(strategy_true_label_list)
        chat_trues = torch.tensor(chat_true_label_list)

        print("Detect 하지 못한 risk : \n", risk_fail)
        print("\nNum(risk_fail) : ", len(risk_fail))
        print("Num(normal_fail) : ", normal_fail)

        ############## Performace #################
        strategy_classification_performance = metrics.classification_report(strategy_trues, label_preds)
        chat_pred_classification_performance = metrics.classification_report(chat_trues, label_preds)

        # classification_performace = metrics.classification_report(doc_trues, label_preds, target_names=['risk', 'normal'])
        classification_performance = metrics.classification_report(doc_trues, label_preds)
        micro_F1 = metrics.f1_score(doc_trues, label_preds, average="micro")
        risk_micro_F1 = metrics.f1_score(doc_trues, label_preds, pos_label=1, average="binary")

        # p = np.log(3) * (1 / (np.median(sents_len) - 1))
        MESSAGE_WITH_HALF_PENALTY = 90
        # p = np.log(3)/(MESSAGE_WITH_HALF_PENALTY-1)

        Latency_F1 = metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p_list) * metrics.f1_score(
            doc_trues, label_preds, pos_label=1, average="binary"
        )

        wandb.log(
            {
                "micro_F1": micro_F1,
                "Latency_F1": Latency_F1,
                # "ERDE": ERDE,
                "speed": metrics_test.latency_weight(doc_trues, label_preds, early_stop_len_utt_list, p),
            }
        )
        print(f"classification_performance: {classification_performance}")
        print(f"strategy_classification_performance: {strategy_classification_performance}")
        print(f"chat_pred_classification_performance: {chat_pred_classification_performance}")
        return (
            classification_performance,
            micro_F1,
            risk_micro_F1,
            Latency_F1,
            # ERDE,
            early_stop_len_utt_list,
            doc_trues,
            label_preds,
        )

    def get_risk_num(self, queue):
        num_risk = len([1 for label in queue if label == "risk"])
        return num_risk
