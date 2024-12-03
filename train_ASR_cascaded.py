import more_itertools as mit
import json
import os
from transformers import BertTokenizer
from tqdm import tqdm
import torch

from Preprocess.dataCollect import get_annotated_data
from trainer import HateSpeechTrainer
from src.models.ModelFactory import ModelFactory
from src.data.AudioHateXplain_datamodule import AudioHateXplainDataModule

# change parameter(att_lam) for attention loss
att_lam_list = [0.001, 0.01, 0.1, 1, 10, 100]

# att_lam_list = [0.001, 0.01, 0.1] # GPU: 0
att_lam_list = [1, 10, 100] # GPU: 1

model_config_prefix = "/home/jinmyeong/code/AudioHateXplain/best_model_json"

# ASR_cascaded

# param
new_model_config_for_cascadedASR_file = f"BERT_train_audioHateXplainASR_F1_w_att_lam=1"
with open(os.path.join(model_config_prefix, new_model_config_for_cascadedASR_file + ".json")) as f:
    ASR_cascaded_model_config = json.load(f)
    params = ASR_cascaded_model_config

    for key in params:
        if params[key] == "True":
            params[key] = True
        elif params[key] == "False":
            params[key] = False
        if key in [
            "batch_size",
            "num_classes",
            "hidden_size",
            "supervised_layer_pos",
            "num_supervised_heads",
            "random_seed",
            "max_length",
        ]:
            if params[key] != "N/A":
                params[key] = int(params[key])

        if (key == "weights") and (params["auto_weights"] == False):
            params[key] = ast.literal_eval(params[key])

    # change in logging to output the results to neptune
    params["logging"] = "local"
    params["device"] = "cuda"
    params["best_params"] = False

    if torch.cuda.is_available() and params["device"] == "cuda":
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
    else:
        print("Since you dont want to use GPU, using the CPU instead.")
        device = torch.device("cpu")

    dict_data_folder = {
        "2": {"data_file": "Data/dataset.json", "class_label": "Data/classes_two.npy"},
        "3": {"data_file": "Data/dataset.json", "class_label": "Data/classes.npy"},
    }

    # Few handy keys that you can directly change.
    params["variance"] = 1
    # params["epochs"] = 5
    params["to_save"] = True
    params["num_classes"] = 2
    # params["data_file"] = 'Data/new_dataset.json'
    params["class_names"] = dict_data_folder[str(params["num_classes"])]["class_label"]
    if params["num_classes"] == 2 and (params["auto_weights"] == False):
        params["weights"] = [1.0, 1.0]


with open(os.path.join(model_config_prefix, new_model_config_for_cascadedASR_file + ".json"), "w") as json_file:
    json.dump(params, json_file, indent=4)

# dataset
dataset = AudioHateXplainDataModule(params=ASR_cascaded_model_config)

# tokenizer

# model
model = ModelFactory.get_model(model_type="SC_weighted_BERT", params=params)

# trainer
trainer = HateSpeechTrainer(param=params,
                            model_type=new_model_config_for_cascadedASR_file,
                            train_dataset=dataset['train'],
                            val_dataset=dataset['validation'],
                            test_dataset=dataset['test'],
                            model=model,
                            tokenizer=tokenizer)
trainer.train()

