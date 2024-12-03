import jiwer
import json

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

file_path = '/home/jinmyeong/code/hts/classifier/Data/audio_rationales_data.json'
with open(file_path, "r") as f:
    audio_rationales_data = json.load(f)
with open('/home/jinmyeong/code/hts/ASR/Data/ASR/whisperX_ASR_transcription_v4-HateSpeech-all-multispeaker.json', 'r') as fp:
    transcript_dict = json.load(fp)
with open("/home/jinmyeong/code/hts/classifier/Data/spoken_humanRecords_post_id_divisions.json", 'r') as fp:
    post_id_division = json.load(fp)

# id_list = post_id_division['train'] + post_id_division['val'] + post_id_division['test']
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

        len_num = len(gold_rationales)
        gold_rationale = []

        for r in gold_rationales:
            if len(gold_rationale) == 0:
                gold_rationale += r
            else:
                for idx, rationale in enumerate(gold_rationale):
                    gold_rationale[idx] += r[idx]
        gold_rationale = [1 if i // len(gold_rationales) == 0.5 else int(i // len(gold_rationales)) for i in gold_rationale]
        rationale_word_wer = get_rationale_word_wer(gold_text_lower, transcript_lower, gold_rationale)
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

file_path = f"/home/jinmyeong/code/hts/classifier/Data/sorted_test_dataset_with_WER.json"
with open(file_path, "w") as f:
    json.dump(sorted_test_dataset_with_WER, f)   