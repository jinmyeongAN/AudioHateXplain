import json
import jiwer


def get_transcript_rationale(gold_audioHateXplain, transcript_dict, post_id_divisions=None):
    # id_list = post_id_divisions['train'] + post_id_divisions['val'] + post_id_divisions['test']
    id_list = list(gold_audioHateXplain.keys())

    transcript_rationales_dict = {}

    for test_id in id_list:
        if gold_audioHateXplain.get(test_id):
            audio_datapoint = gold_audioHateXplain[test_id]

            audio_id = audio_datapoint['audio_id']
            gold_text = " ".join(audio_datapoint['sentence'])
            gold_rationales = audio_datapoint['rationale']
            
            # get transcript_word_list
            transcript_word_dict_list = []
            for transcript_dict_element_dict in transcript_dict[test_id]:
                if transcript_dict[test_id] == "ERROR": # ASR로 인식 못하는 경우
                    transcript_word_list, transcript_text = [], ''
                    continue
                transcript_word_dict_list += transcript_dict_element_dict['words']
    
            if transcript_dict[test_id] != "ERROR":    
                transcript_word_list = [word_dict['word'] for word_dict in transcript_word_dict_list]
                # get transcript_text
                transcript_text = " ".join(transcript_word_list)
            
            # transcript_rationales 0으로 초기화
            len_transcript_word = len(transcript_word_list)
            len_transcript_rationales = len(gold_rationales)

            transcript_rationales = [len_transcript_word * [0] for _ in range(len_transcript_rationales)]

            error = jiwer.process_words(gold_text, transcript_text)
            alignments = error.alignments[0]

            for alignment in alignments:
                align_type = alignment.type
                ref_start_idx = alignment.ref_start_idx
                ref_end_idx = alignment.ref_end_idx
                hyp_start_idx = alignment.hyp_start_idx
                hyp_end_idx = alignment.hyp_end_idx
                

                # align이 맞을 경우, ref의 rationale을 hyp에 넣는다.
                for idx, gold_rationale in enumerate(gold_rationales):
                    if align_type=='equal':
                        transcript_rationales[idx][hyp_start_idx:hyp_end_idx] = gold_rationale[ref_start_idx:ref_end_idx] #if len(gold_rationale[ref_start_idx:ref_end_idx]) != ref_end_idx - ref_start_idx
        
            transcript_rationales_dict[audio_id] = transcript_rationales    
    return transcript_rationales_dict

def get_hateXplain_style_dataset(audioHateXplain):
    dataset = {}

    for doc_id in audioHateXplain:
        post_id = doc_id

        annotators = []
        for idx, _ in enumerate(audioHateXplain[doc_id]['rationale']):
            annotators.append({"label": f"{audioHateXplain[doc_id]['label']}", "annotator_id": idx + 1, "target": "cascasian"})
            
        rationales = []
        post_tokens = []
        if audioHateXplain[doc_id]['intervals']:
            for intervals in audioHateXplain[doc_id]['intervals']:
                post_token = audioHateXplain[doc_id]['intervals'][intervals]['text']
                post_tokens.append(post_token)

            rationales = audioHateXplain[doc_id]['rationale']

        dataset[doc_id] = {"post_id": post_id, "annotators": annotators, "rationales": rationales, "post_tokens": post_tokens}
    return dataset
# ---------------------------------------------------------------
def get_majority_rationale(rationale_list):
    majority_rationale = []
    for idx in range(len(rationale_list[0])):
        candidated_len = 0
        candidated_sum = 0
        for rationale in rationale_list:
            try:
                candidated_sum += rationale[idx]
                candidated_len += 1
            except:
                candidated_sum += 0
                candidated_len += 0

        majority_ratinale_element = 1 if candidated_sum / candidated_len >= 0.5 else 0      
        majority_rationale.append(majority_ratinale_element)
    
    return majority_rationale

def get_intervals(hatemm_whispher, doc_id, rationale_list):
    intervals = {}
    word_dict_list = []

    majority_rationale = get_majority_rationale(rationale_list)
    hatemm_whispher_list = hatemm_whispher[doc_id]
    
    for hatemm in hatemm_whispher_list:
        word_dict_list += hatemm['words']
        
    for idx, word_dict in enumerate(word_dict_list):
        word = word_dict['word']
        start = word_dict['start'] if word_dict.get('start') else None
        end = word_dict['end'] if word_dict.get('end') else None
        

        intervals[f"intervals [{idx+1}]"] = {"xmin": start,
                                             "xmax": end,
                                             "text": word,
                                             "rationale": majority_rationale[idx]}
    return intervals

def get_audioHateXplain_from_transcript(ASR_transcription,  transcript_rationale_dict, gold_hatemm_audioHateXplain):
    hatemm_audioHateXplain = {}

    for doc_id in ASR_transcription:
        if transcript_rationale_dict.get(doc_id) == None:
            continue
        rationale_list = transcript_rationale_dict[doc_id]
        if ASR_transcription[doc_id] == "ERROR":
            intervals = None
            sentence = None
        else:
            intervals = get_intervals(ASR_transcription, doc_id, rationale_list)
            word_list = [intervals[key]['text'] for key in intervals]
            sentence = " ".join(word_list)
        label = gold_hatemm_audioHateXplain[doc_id]['label']
        hatemm_audioHateXplain[doc_id] = {"audio_id": doc_id,
                                            "sentence": sentence,
                                            "intervals": intervals,
                                            "rationale": rationale_list,
                                            "label": label}
    return hatemm_audioHateXplain


def get_cacacadedDataset_from_audioHateXplain_and_ASR(gold_audioHateXplain_path, ASR_transcription_path, data_name="hatemm"):
    # Load gold_hatemm_audioHateXplain dataset
    gold_hatemm_audioHateXplain_path = gold_audioHateXplain_path
    with open(gold_hatemm_audioHateXplain_path, "r") as f: # from "Data" folder
        gold_hatemm_audioHateXplain = json.load(f)

    # Load ASR transcript of gold_hatemm
    whisperX_ASR_transcription_hatemm_path = ASR_transcription_path
    with open(whisperX_ASR_transcription_hatemm_path, 'r') as fp:
        hatemm_transcript = json.load(fp)

    # load hatemm post_id_division
    # hatemm_post_id_divisions = {'test':[], 'val':[], 'train':[]}
    # hatemm_post_id_divisions['test'] = list(gold_hatemm_audioHateXplain.keys())

    # find transcript rationale
    hatemm_transcript_rationale = get_transcript_rationale(gold_audioHateXplain=gold_hatemm_audioHateXplain,
                                                        transcript_dict=hatemm_transcript, 
                                                        )

    # Change ASR transcript into audio_hateXplain style
    hatemm_transcript_audioHateXplain = get_audioHateXplain_from_transcript(ASR_transcription=hatemm_transcript,
                                                                            transcript_rationale_dict=hatemm_transcript_rationale,
                                                                            gold_hatemm_audioHateXplain=gold_hatemm_audioHateXplain)

    # get hatemm_transcript dataset (HateXplain dataset.json style)
    hatemm_transcript_dataset = get_hateXplain_style_dataset(audioHateXplain=hatemm_transcript_audioHateXplain)


    # hatemm_post_id_divisions_path = f"/home/jinmyeong/code/hts/classifier/Data/{data_name}_post_id_divisions.json"
    # with open(hatemm_post_id_divisions_path, 'w') as fp:
    #     json.dump(hatemm_post_id_divisions, fp)

    file_path = f"/home/jinmyeong/code/hts/classifier/Data/{data_name}_dataset.json"
    with open(file_path, "w") as f:
        json.dump(hatemm_transcript_dataset, f)   
    print()


# ground truth ASR hateMM
dataset_type = "hatemm"
get_cacacadedDataset_from_audioHateXplain_and_ASR(gold_audioHateXplain_path=f"/home/jinmyeong/code/hts/classifier/Data/groudTruth_{dataset_type}_audioHateXplain.json",
                                                  ASR_transcription_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/groundTruth_ASR_transcription_{dataset_type}.json",
                                                  data_name=f"groundTruth_{dataset_type}")

# HateMM dataset
dataset_type = "hatemm"
whisper_size = ["tiny", "base", "small", "medium", "large", "large-v2"]
for whisper_type in whisper_size:
    get_cacacadedDataset_from_audioHateXplain_and_ASR(gold_audioHateXplain_path=f"/home/jinmyeong/code/hts/classifier/Data/groudTruth_{dataset_type}_audioHateXplain.json",
                                                  ASR_transcription_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/merged_whisperX-{whisper_type}_ASR_transcription_{dataset_type}.json",
                                                  data_name=f"whisperX-{whisper_type}_{dataset_type}")

# ground truth ASR
dataset_type = "audioHateXplain"
get_cacacadedDataset_from_audioHateXplain_and_ASR(gold_audioHateXplain_path=f"/home/jinmyeong/code/hts/classifier/Data/groudTruth_{dataset_type}.json",
                                                  ASR_transcription_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/groundTruth_ASR_transcription_audioHateXplain.json",
                                                  data_name=f"groundTruth_{dataset_type}")

# audioHateXplain + various Whisper model
# dataset_type = "audioHateXplain"
# whisper_size = ["tiny", "base", "small", "medium", "large", "large-v2"]
# for whisper_type in whisper_size:
#     get_cacacadedDataset_from_audioHateXplain_and_ASR(gold_audioHateXplain_path=f"/home/jinmyeong/code/hts/classifier/Data/groudTruth_{dataset_type}.json",
#                                                   ASR_transcription_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/whisper-{whisper_type}_ASR_transcription_{dataset_type}.json",
#                                                   data_name=f"whisper-{whisper_type}_{dataset_type}")

# audioHateXplain + various WhisperX model
dataset_type = "audioHateXplain"
whisper_size = ["tiny", "base", "small", "medium", "large", "large-v2"]
for whisper_type in whisper_size:
    get_cacacadedDataset_from_audioHateXplain_and_ASR(gold_audioHateXplain_path=f"/home/jinmyeong/code/hts/classifier/Data/groudTruth_{dataset_type}.json",
                                                  ASR_transcription_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/merged_whisperX-{whisper_type}_ASR_transcription_{dataset_type}.json",
                                                  data_name=f"whisperX-{whisper_type}_{dataset_type}")
