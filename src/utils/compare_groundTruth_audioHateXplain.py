import json
import pickle
from sklearn.metrics import accuracy_score, f1_score


# load new_dataset
new_dataset_path = "/home/jinmyeong/code/hts/classifier/Data/new_dataset.json"
with open(new_dataset_path, "r") as f: # from "Data" folder
    new_dataset = json.load(f)

# load /home/jinmyeong/code/hts/classifier/Data/groundTruth_audioHateXplain_dataset.json
WER_audioHateXplain_dataset_path = "/home/jinmyeong/code/hts/classifier/Data/whisperX_ASR_transcription_v4_audioHateXplain_dataset.json"
with open(WER_audioHateXplain_dataset_path, "r") as f: # from "Data" folder
    WER_audioHateXplain_dataset = json.load(f)

# load /home/jinmyeong/code/hts/classifier/Data/audio_rationales_data.json
audio_rationales_data_path = "/home/jinmyeong/code/hts/classifier/Data/audio_rationales_data.json"
with open(audio_rationales_data_path, "r") as f: # from "Data" folder
    audio_rationales_data = json.load(f)

groundTruth_audioHateXplain_path = "/home/jinmyeong/code/hts/classifier/Data/groudTruth_audioHateXplain.json"
with open(groundTruth_audioHateXplain_path, "r") as f: # from "Data" folder
    groundTruth_audioHateXplain = json.load(f)

# Load ASR transcript of gold_hatemm
whisperX_ASR_transcription_hatemm_path = "/home/jinmyeong/code/hts/ASR/Data/ASR/whisperX-large-v2_ASR_transcription_audioHateXplain.json"
with open(whisperX_ASR_transcription_hatemm_path, 'r') as fp:
    whisperX_transcript = json.load(fp)

# Load ASR transcript of gold_hatemm
whisperX_ASR_transcription_v4_audioHateXplain_dataset_path = "/home/jinmyeong/code/hts/classifier/Data/whisperX_ASR_transcription_v4_audioHateXplain_dataset.json"
with open(whisperX_ASR_transcription_v4_audioHateXplain_dataset_path, 'r') as fp:
    whisperX_ASR_transcription_v4_audioHateXplain_dataset = json.load(fp)


with open("/home/jinmyeong/code/hts/classifier/Data/Total_new_whisperX-large-v2_audioHateXplain_ASR_text_bert_softmax_1_128_2/test_data.pickle", "rb") as f:
    new_dataset_test = pickle.load(f)

with open(f'/home/jinmyeong/code/hts/classifier/Data/Total_WER_whisperX-large-v2_audioHateXplain_ASR_text_bert_softmax_1_128_2/test_data.pickle', "rb") as f:
    WER_test = pickle.load(f)


for e in new_dataset_test:
    key = e[3]
    label = e[2]
    text = e[0]
    WER_label = None
    WER_text = None
    for i in WER_test:
        WER_key = i[3]
        if WER_key == key:
            WER_label = i[2]
            WER_text = i[0]
            break
    
    if WER_label:
        if WER_label != label:
            print("Different label")
    else:
        print("No key in WER")

    if WER_text:
        if "".join([str(i) for i in WER_text]) != "".join([str(i) for i in text]):
            print("Different text")
    else:
        print("No key in WER")    


with open(f'/home/jinmyeong/code/hts/classifier/Result/Result_GT_labels.json', "r") as f:
    GT_labels = json.load(f)
GT_pred_labels = GT_labels["pred_labels"]
GT_true_labels = GT_labels["true_labels"]

with open(f'/home/jinmyeong/code/hts/classifier/Result/Result_new_dataset_labels.json', "r") as f:
    new_dataset_labels = json.load(f)
new_dataset_pred_labels = new_dataset_labels["pred_labels"]
new_dataset_true_labels = new_dataset_labels["true_labels"]

for idx, _ in enumerate(GT_pred_labels):
    if GT_pred_labels[idx] != new_dataset_pred_labels[idx]:
        print(idx)

for idx, _ in enumerate(GT_true_labels):
    if GT_true_labels[idx] != new_dataset_true_labels[idx]:
        print(idx)

GT_f1 = f1_score(GT_true_labels, GT_pred_labels, average="macro")
new_dataset_f1 = f1_score(new_dataset_true_labels, new_dataset_pred_labels, average="macro")



for idx, key in enumerate(groundTruth_audioHateXplain_dataset):
    if " ".join(new_dataset[key]['post_tokens']) != " ".join(whisperX_ASR_transcription_v4_audioHateXplain_dataset[key]['post_tokens']):
        print(f"id: {key}")
        print(f"new_dataset: {' '.join(new_dataset[key]['post_tokens'])}")
        print(f"groundTruth: {' '.join(whisperX_ASR_transcription_v4_audioHateXplain_dataset[key]['post_tokens'])}")
        print()
print()