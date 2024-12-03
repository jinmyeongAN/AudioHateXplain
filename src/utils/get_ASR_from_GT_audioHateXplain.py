import json


def get_cacacadedDataset_from_audioHateXplain_and_ASR(gold_audioHateXplain_path, ASR_transcription_path):
    # Load gold_hatemm_audioHateXplain dataset
    gold_hatemm_audioHateXplain_path = gold_audioHateXplain_path
    with open(gold_hatemm_audioHateXplain_path, "r") as f: # from "Data" folder
        gold_hatemm_audioHateXplain = json.load(f)

    # Load ASR transcript of gold_hatemm
    whisperX_ASR_transcription_hatemm_path = ASR_transcription_path
    with open(whisperX_ASR_transcription_hatemm_path, 'r') as fp:
        hatemm_transcript = json.load(fp)  
    
    groundTruth_ASR = {}
    for doc_id, value in gold_hatemm_audioHateXplain.items():
        start, end = 0, 0
        texts = []
        words = []

        for idx, _ in enumerate(list(value['intervals'].keys())):
            interval = value['intervals'][f"intervals [{idx + 1}]"]
            xmin, xmax, text = interval['xmin'], interval['xmax'], interval['text']
            
            texts.append(text)
            word_element = {'word': text, 'start': xmin, 'end': xmax}
            words.append(word_element)

        
        groundTruth_ASR[doc_id] = [{'start': start, 'end': end, 'text': " " + " ".join(texts), 'words': words}]
    return groundTruth_ASR

# dataset_type = "hatemm"
# whisper_type = "large-v2"

# groundTruth_ASR = get_cacacadedDataset_from_audioHateXplain_and_ASR(gold_audioHateXplain_path=f"/home/jinmyeong/code/hts/classifier/Data/groudTruth_{dataset_type}_audioHateXplain.json",
#                                                   ASR_transcription_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/merged_whisperX-{whisper_type}_ASR_transcription_{dataset_type}.json")

# file_path = f"/home/jinmyeong/code/hts/ASR/Data/ASR/groundTruth_ASR_transcription_{dataset_type}.json"
# with open(file_path, "w") as f:
#     json.dump(groundTruth_ASR, f)   
# print()

dataset_type = "audioHateXplain"
whisper_type = "large-v2"

groundTruth_ASR = get_cacacadedDataset_from_audioHateXplain_and_ASR(gold_audioHateXplain_path=f"/home/jinmyeong/code/hts/classifier/Data/groudTruth_{dataset_type}.json",
                                                  ASR_transcription_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/merged_whisperX-{whisper_type}_ASR_transcription_{dataset_type}.json")

file_path = f"/home/jinmyeong/code/hts/ASR/Data/ASR/groundTruth_ASR_transcription_audioHateXplain.json"
with open(file_path, "w") as f:
    json.dump(groundTruth_ASR, f)   
print()