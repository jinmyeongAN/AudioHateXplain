from typing import Any, Dict, List, Optional, Tuple

import logging
import os
import random
import sys
import warnings
import json

import datasets
import evaluate
import hydra
import lightning as L
import numpy as np
import rootutils
import torch
import transformers
from datasets import Dataset, load_from_disk
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    PretrainedConfig,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint

from src.models import ModelFactory

# from src.models.actorCritic_policy_module import ActorCriticGPT2, get_state, train
# from src.models.REINFORCE_policy_module import ActorCriticGPT2, get_state, train
# from src.models.REINFORCE_RoBERTa_module import ActorCriticRoBERTa, get_state, train
from src.models.onlyREINFORCE_RoBERTa_module import (
    ActorCriticRoBERTa,
    get_groundTruth_state,
    get_state,
    get_states_for_policy,
    rl_eval,
    train,
)
from src.trainer import TrainerFactory

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import pickle
from random import shuffle

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


log = RankedLogger(__name__, rank_zero_only=True)
logger = logging.getLogger(__name__)


# @task_wrapper
def run(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # DataTrainingArguments
    log.info(f"Instantiating DataTrainingArguments <{cfg.args.data_args._target_}>")
    data_args = hydra.utils.instantiate(cfg.args.data_args)

    # ModelArguments
    log.info(f"Instantiating ModelArguments <{cfg.args.model_args._target_}>")
    model_args = hydra.utils.instantiate(cfg.args.model_args)

    # TrainingArguments
    log.info(f"Instantiating TrainingArguments <{cfg.args.training_args._target_}>")
    training_args = hydra.utils.instantiate(cfg.args.training_args)

    # # Trainer Factory
    # log.info(f"Instantiating Trainer Factory <{cfg.trainer._target_}>")
    # trainer_factory = hydra.utils.instantiate(cfg.trainer)
    trainerFactory = TrainerFactory()
    Trainer = trainerFactory.get_trainer(trainer_type=model_args.trainer_type)

    # Config
    # log.info(f"Instantiating Config <{cfg.config._target_}>")
    # config = hydra.utils.instantiate(cfg.config)

    # Tokenizer
    log.info(f"Instantiating Tokenizer <{cfg.tokenizer._target_}>")
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    tokenizer.truncation_side = "left"

    # Define new special tokens
    new_tokens = ["[predator]", "[victim]", "[PI]", "[G]", "[A]", "[O]"]
    # Add new tokens to the tokenizer
    num_added_toks = tokenizer.add_tokens(new_tokens)
    print("We have added", num_added_toks, "tokens")
    # Resize model embeddings to account for new tokens

    # Resize model embeddings to account for new tokens
    # model.resize_token_embeddings(len(tokenizer))

    # download the dataset.
    log.info(f"Instantiating Tokenizer <{cfg.data._target_}>")
    raw_datasets = hydra.utils.instantiate(cfg.data).dataset

    os.environ["WANDB_PROJECT"] = data_args.task_name  # name your W&B project
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # A useful fast method:
    # https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.Dataset.unique
    label_list = raw_datasets["train"].unique("label")
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)

    # Model
    # log.info(f"Instantiating Model <{cfg.model._target_}>")
    # model = hydra.utils.instantiate(cfg.model)
    modelFactory = ModelFactory()

    model = modelFactory.get_model(model_args.model_name_or_path, model_args.ckpt_path, num_labels=num_labels)
    # Resize model embeddings to account for new tokens
    model.resize_token_embeddings(len(tokenizer))
    model.train()

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        # and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
            label_to_id = {v: i + 1 for i, v in enumerate(label_list)}
    else:  # data_args.task_name is not None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    # modify label_to_id
    if len(label_list) > 1:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    else:
        label_to_id = {v: i + 1 for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in label_to_id.items()}
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}
    # elif data_args.task_name is not None and not is_regression:
    #     model.config.label2id = {l: i for i, l in enumerate(label_list)}
    #     model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(result):
        # Tokenize the texts
        # args = (
        #     (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        # )
        # text = examples["text"]
        # result = tokenizer(text, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in result:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in result["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(eval_pred: EvalPrediction):
        f1_metric = evaluate.load("f1")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        result = f1_metric.compute(predictions=predictions, references=labels, average="micro")
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Training
    # if training_args.do_train:
    #     checkpoint = None
    #     if training_args.resume_from_checkpoint is not None:
    #         checkpoint = training_args.resume_from_checkpoint
    #     elif last_checkpoint is not None:
    #         checkpoint = last_checkpoint
    #     train_result = trainer.train(resume_from_checkpoint=checkpoint)
    #     metrics = train_result.metrics
    #     max_train_samples = (
    #         data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    #     )
    #     metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    #     trainer.save_model()  # Saves the tokenizer too for easy upload

    #     trainer.log_metrics("train", metrics)
    #     trainer.save_metrics("train", metrics)
    #     trainer.save_state()

    # 🔥 policy training ==============================================

    train_states = get_states_for_policy(dataset=raw_datasets["train"], isShuffle=True)
    val_states = get_states_for_policy(dataset=raw_datasets["validation"], isShuffle=False)
    test_states = get_states_for_policy(dataset=raw_datasets["test"], isShuffle=False)

    # import random

    random.seed(42)
    sample_size = 690
    neg_test_dataset_idx = [idx for idx, datapoint in enumerate(raw_datasets["test"]) if datapoint['label'] == 0]
    pos_test_dataset_idx = [idx for idx, datapoint in enumerate(raw_datasets["test"]) if datapoint['label'] != 0]
    random_test_dataset_idx =  random.sample(neg_test_dataset_idx, sample_size)+ pos_test_dataset_idx

    negHundred_test_dataset = raw_datasets["test"].filter(lambda example, idx: len(example['text']) >= 100, with_indices=True)
    negHundred_test_states = get_states_for_policy(dataset=negHundred_test_dataset, isShuffle=False)
    
    random_test_dataset = raw_datasets["test"].filter(lambda example, idx: idx in random_test_dataset_idx, with_indices=True)
    random_test_states = get_states_for_policy(dataset=random_test_dataset, isShuffle=False)
    # train_states = get_no_strategy_states_for_policy(dataset=raw_datasets["train"], isShuffle=True)
    # val_states = get_no_strategy_states_for_policy(dataset=raw_datasets["validation"], isShuffle=False)
    # test_states = get_no_strategy_states_for_policy(dataset=raw_datasets["test"], isShuffle=False)
    train_neg_strategy_num = []
    posTrain_neg_strategy_num = []
    negTrain_neg_strategy_num = []

    for train_datapoint in raw_datasets["train"]:
        train_datapoint_stage = train_datapoint['stage']
        binary_stages_list = [1 if "G" in stage or "A" in stage or "R" in stage or "C" in stage or "I" in stage else 0 for stage in train_datapoint_stage]
        num_neg_strategy = len([i for i in binary_stages_list if i == 1])      
        train_datapoint_label = train_datapoint['label']

        if train_datapoint_label == 0:
            train_neg_strategy_num.append(num_neg_strategy)
            negTrain_neg_strategy_num.append(num_neg_strategy)
        elif train_datapoint_label == 1:
            train_neg_strategy_num.append(num_neg_strategy)
            posTrain_neg_strategy_num.append(num_neg_strategy)       

    strategy_num_obj = {"train_neg_strategy_num": train_neg_strategy_num,
                        "posTrain_neg_strategy_num": posTrain_neg_strategy_num,
                        "negTrain_neg_strategy_num": negTrain_neg_strategy_num}
    # with open('/home/jinmyeong/code/eSPD/data/strategy_num_obj.json', 'w') as f:
    #     json.dump(strategy_num_obj, f, indent=2)

    if training_args.do_train:
        train(
            train_states=train_states,
            val_states=val_states, #random_test_states, #val_states, # random_test_states
            model=model,
            tokenizer=tokenizer,
            saved_path=training_args.output_dir,
            task_name=data_args.task_name
        )
    if training_args.do_predict:
        rl_eval(test_states=test_states, model=model, tokenizer=tokenizer, test_dataset=raw_datasets["test"])

    # ==============================================🔥 policy training

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

    return


@hydra.main(version_base="1.3", config_path="./configs", config_name="train_RL_onlyPosAnnSFT.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    run(cfg)

    return


if __name__ == "__main__":
    main()
