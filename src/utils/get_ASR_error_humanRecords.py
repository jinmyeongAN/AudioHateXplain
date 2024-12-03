import jiwer
import json

import pickle

groundTruth_rationale_path = "/home/jinmyeong/code/hts/classifier/Data/hateXplain_with_rationale.pickle"
# load pickle
with open(file=f"{groundTruth_rationale_path}", mode='rb') as f:
    groundTruth_rationale=pickle.load(f)

def calculate_iou(interval1, interval2):
    t1, t2 = interval1
    t3, t4 = interval2

    # Calculate the overlap
    overlap_start = max(t1, t3)
    overlap_end = min(t2, t4)
    overlap_duration = max(0, overlap_end - overlap_start)

    # Calculate the union
    union_start = min(t1, t3)
    union_end = max(t2, t4)
    union_duration = union_end - union_start

    # Calculate IoU
    iou = overlap_duration / union_duration if union_duration != 0 else 0
    return iou

def get_rationale_word_wer(reference, hypothesis, reference_rationale):
    output = jiwer.process_words(reference, hypothesis)
    all_alignments = []
    alignment_list = output.alignments

    trues = 0
    falses = 0

    for align in alignment_list:
        all_alignments += align

    for idx, rationale in enumerate(reference_rationale):
        if rationale == 1:
            for align in all_alignments:
                if idx >= align.ref_start_idx and idx < align.ref_end_idx:
                    if align.type == 'equal':
                        trues += 1
                    else:
                        falses += 1
    error = falses / (trues + falses) if trues + falses > 0 else None
    return error    


def get_WER(reference, hypothesis):
    error = jiwer.wer(reference, hypothesis)
    
    output = jiwer.process_words(reference, hypothesis)
    error = output.wer
    return error

def get_CER(reference, hypothesis):
    error = jiwer.cer(reference, hypothesis)

    output = jiwer.process_characters(reference, hypothesis)
    error = output.cer

    return error
Gold_model_list = ["whisperX-tiny", "whisperX-base" ,"whisperX-small", 
                  "whisperX-medium", "whisperX-large-v2",]
file_path = "/home/jinmyeong/code/hts/ASR/human_recording/groundTruth_humanRecords.json"
with open(file_path, "r") as f:
    audio_rationales_data = json.load(f)
with open("/home/jinmyeong/code/hts/ASR/Data/ASR/merged_whisperX-medium_ASR_transcription_humanRecords.json", 'r') as fp:
    transcript_dict = json.load(fp)
with open("/home/jinmyeong/code/hts/classifier/Data/spoken_humanRecords_post_id_divisions.json", 'r') as fp:
    post_id_division = json.load(fp)

spoken_id_list = list(transcript_dict.keys())
word_iou_list = []
for spoken_id in spoken_id_list:
    ASR_word_list = transcript_dict[spoken_id][0]['words']
    gold_word_list = [audio_rationales_data[spoken_id]['intervals'][f"intervals [{idx+1}]"] for idx, _ in enumerate(list(audio_rationales_data[spoken_id]['intervals'].keys()))]
    gold_word_list = [{"start": i['xmin'], "end": i['xmax'], 'word': i['text']}for i in gold_word_list]

    ASR_text = " ".join([w["word"] for w in ASR_word_list]).strip().lower()
    gold_text = " ".join([w["word"] for w in gold_word_list]).strip().lower()

    output = jiwer.process_words(reference=gold_text, hypothesis=ASR_text)
    alignmentChunks = output.alignments[0]
    equal_alignmentChunk = [{'ref_idx_range': (a.ref_start_idx, a.ref_end_idx), 'hyp_idx_range': (a.hyp_start_idx, a.hyp_end_idx)} for a in alignmentChunks if a.type == 'equal']

    for equal_align in equal_alignmentChunk:
        ref_start_idx, ref_end_idx = equal_align['ref_idx_range']
        hyp_start_idx, hyp_end_idx = equal_align['hyp_idx_range']

        ref_idx_list = list(range(ref_start_idx, ref_end_idx))
        hyp_idx_list = list(range(hyp_start_idx, hyp_end_idx))

        for idx, _ in enumerate(ref_idx_list):
            ref_word = gold_word_list[ref_idx_list[idx]]
            hyp_word = ASR_word_list[hyp_idx_list[idx]]

            ref_idx = ref_idx_list[idx]
            hyp_idx = hyp_idx_list[idx]

            iou = calculate_iou(interval1=(gold_word_list[ref_idx]['start'], gold_word_list[ref_idx]['end']), interval2=(ASR_word_list[hyp_idx]['start'],ASR_word_list[hyp_idx]['end']))
            
            word_iou_list.append(iou)
            # if iou < 0.2:
            #     print(iou)
            #-----------------------------

            # groundTruth interval 넣은 transcript 만들기
            transcript_dict[spoken_id][0]['words'][hyp_idx]['start'],transcript_dict[spoken_id][0]['words'][hyp_idx]['end'] = gold_word_list[ref_idx]['start'], gold_word_list[ref_idx]['end']
            
            #-----------------------------
    # for idx, word in enumerate(gold_word_list):
    #     if idx < len(gold_word_list) and idx < len(ASR_word_list) and gold_word_list[idx]['text'] == ASR_word_list[idx]['word']:
    #         iou = calculate_iou(interval1=(gold_word_list[idx]['start'], gold_word_list[idx]['end']), interval2=(ASR_word_list[idx]['start'],ASR_word_list[idx]['end']))
    #         word_iou_list.append(iou)
    #         if iou < 0.7:
    #             print(iou)
# id_list = post_id_division['train'] + post_id_division['val'] + post_id_division['test']
#-----------------------------

# groundTruth interval 넣은 transcript 만들기
file_path = f"/home/jinmyeong/code/hts/ASR/Data/ASR/merged_whisperX-large-v2_ASR_transcription_humanRecords_goldTimestamping.json"
with open(file_path, "w") as f:
    json.dump(transcript_dict, f) 
#-----------------------------



# word_align_iou= sum(word_iou_list)/len(word_iou_list)
id_list = post_id_division['test']
transcript_rationales_dict = {}

WER_list = []
CER_list = []

rationale_word_wer_list = []

test_dataset_with_WER = []

for test_id in id_list:
    if audio_rationales_data.get(test_id):
        audio_datapoint = audio_rationales_data[test_id]

        audio_id = audio_datapoint['audio_id']
        gold_word_list = audio_datapoint['sentence']
        gold_text = " ".join(gold_word_list)
        gold_rationales = audio_datapoint['rationale']

        full_text_list = [word_dict['text'] for word_dict in transcript_dict[test_id]]
        full_text = " ".join(full_text_list).strip()

        transcript = full_text
        # transcript = transcript_dict[test_id][0]['text'][1:]
        # transcript_word_list = [word['word'] for word in transcript_dict[test_id][0]['words']]



        # transcript_rationales 0으로 초기화
        # len_transcript_word = len(transcript_word_list)
        len_transcript_rationales = len(gold_rationales)

        # transcript_rationales = [len_transcript_word * [0] for _ in range(len_transcript_rationales)]

        error = jiwer.process_words(gold_text, transcript)
        alignments = error.alignments[0]

        gold_text_lower = gold_text.lower().replace('.', '').replace(',', '').replace('<', '').replace('>', '').replace('?', '').replace('-', ' ')
        transcript_lower = transcript.lower().replace('.', '').replace(',', '').replace('<', '').replace('>', '').replace('?', '').replace('-', ' ')

        wer = get_WER(gold_text_lower, transcript_lower)
        cer = get_CER(gold_text_lower, transcript_lower)

        rationale_word_wer = get_rationale_word_wer(gold_text_lower, transcript_lower, gold_rationales[0])

        # if wer >= 0.25:
        #     print(gold_text_lower)
        #     print("-" * 20)
        #     print(transcript_lower)
        #     print("=" * 20)


        WER_list.append(wer)
        CER_list.append(cer)
    
        rationale_word_wer_list.append(rationale_word_wer)

        # measure each datapoint's WER
        data_with_WER = {"id": test_id, "wer": wer}
        test_dataset_with_WER.append(data_with_WER)

print(f"WER: {sum(WER_list) / len(WER_list)}")
print(f"CER: {sum(CER_list) / len(CER_list)}")

rationale_word_wer_list = [i for i in rationale_word_wer_list if i != None]
print(f"rationale_word_WER: {sum(rationale_word_wer_list) / len(rationale_word_wer_list)}")
print()


sorted_test_dataset_with_WER = sorted(test_dataset_with_WER, key=lambda datapoint: datapoint["wer"])

file_path = f"/home/jinmyeong/code/hts/classifier/Data/sorted_test_humanRecords_dataset_with_WER.json"
with open(file_path, "w") as f:
    json.dump(sorted_test_dataset_with_WER, f)   