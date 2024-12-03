import json
import jiwer


def get_full_text_from_ASR(ASR_transcript):
    full_text_from_ASR = {}
    for key, value in ASR_transcript.items():
        full_text_list = [word_dict['text'] for word_dict in value] if value != "ERROR" else ['']
        full_text = " ".join(full_text_list).strip()

        full_text_from_ASR[key] = full_text
    return full_text_from_ASR

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

def get_WER_from_gold_ASR_transcript(gold_audioHateXplain_path, ASR_result_path):
    # load hatemm gold transcript
    gold_hatemm_audioHateXplain_path = gold_audioHateXplain_path
    with open(gold_hatemm_audioHateXplain_path, "r") as f:
        gold_hatemm_audioHateXplain = json.load(f)

    # load hatemm ASR transcript
    with open(ASR_result_path, 'r') as fp:
        ASR_transcription_hatemm = json.load(fp)   

    gold_hatemm_transcript_dict = {}
    ASR_hatemm_transcript_dict = {}

    WER_list = []
    CER_list = []

    # 1. Get gold transcript
    for key, value in gold_hatemm_audioHateXplain.items():
        gold_hatemm_transcript_dict[key] = " ".join(value['sentence'])

    # 2. Get ASR transcript
    ASR_hatemm_transcript_dict = get_full_text_from_ASR(ASR_transcription_hatemm) 


    for key in gold_hatemm_transcript_dict:
        gold_text = gold_hatemm_transcript_dict[key]
        transcript = ASR_hatemm_transcript_dict[key]

        error = jiwer.process_words(gold_text, transcript)
        alignments = error.alignments[0]

        gold_text_lower = gold_text.lower().replace('.', '').replace(',', '').replace('<', '').replace('>', '').replace('?', '').replace('-', ' ')
        transcript_lower = transcript.lower().replace('.', '').replace(',', '').replace('<', '').replace('>', '').replace('?', '').replace('-', ' ')

        wer = get_WER(gold_text_lower, transcript_lower)
        cer = get_CER(gold_text_lower, transcript_lower)

        # if wer >= 0.25:
        #     print(f"Gold: {gold_text_lower}")
        #     print("-" * 20)
        #     print(f"ASR: {transcript_lower}")
        #     print("=" * 20)


        WER_list.append(wer)
        CER_list.append(cer)


    print(f"WER: {sum(WER_list) / len(WER_list)}")
    print(f"CER: {sum(CER_list) / len(CER_list)}")

def get_WER_from_gold_ASR_transcript_whisper(gold_audioHateXplain_path, ASR_result_path):
    # load hatemm gold transcript
    gold_hatemm_audioHateXplain_path = gold_audioHateXplain_path
    with open(gold_hatemm_audioHateXplain_path, "r") as f:
        gold_hatemm_audioHateXplain = json.load(f)

    # load hatemm ASR transcript
    with open(ASR_result_path, 'r') as fp:
        ASR_transcription_hatemm = json.load(fp)   

    gold_hatemm_transcript_dict = {}
    ASR_hatemm_transcript_dict = {}

    WER_list = []
    CER_list = []

    # 1. Get gold transcript
    for key, value in gold_hatemm_audioHateXplain.items():
        gold_hatemm_transcript_dict[key] = " ".join(value['sentence'])

    # 2. Get ASR transcript
    for key, value in ASR_transcription_hatemm.items():
        ASR_transcription_hatemm[key] = value.strip()
    ASR_hatemm_transcript_dict = ASR_transcription_hatemm #get_full_text_from_ASR(ASR_transcription_hatemm) 


    for key in gold_hatemm_transcript_dict:
        gold_text = gold_hatemm_transcript_dict[key]
        transcript = ASR_hatemm_transcript_dict[key]

        error = jiwer.process_words(gold_text, transcript)
        alignments = error.alignments[0]

        gold_text_lower = gold_text.lower().replace('.', '').replace(',', '').replace('<', '').replace('>', '').replace('?', '').replace('-', ' ')
        transcript_lower = transcript.lower().replace('.', '').replace(',', '').replace('<', '').replace('>', '').replace('?', '').replace('-', ' ')

        wer = get_WER(gold_text_lower, transcript_lower)
        cer = get_CER(gold_text_lower, transcript_lower)

        # if wer >= 0.25:
        #     print(f"Gold: {gold_text_lower}")
        #     print("-" * 20)
        #     print(f"ASR: {transcript_lower}")
        #     print("=" * 20)


        WER_list.append(wer)
        CER_list.append(cer)


    print(f"WER: {sum(WER_list) / len(WER_list)}")
    print(f"CER: {sum(CER_list) / len(CER_list)}")

# hatemm WhisperX
whisper_size = ["tiny", "base", "small", "medium", "large", "large-v2"]
for whisper_type in whisper_size:

    whisper_hatemm_WER = get_WER_from_gold_ASR_transcript(gold_audioHateXplain_path="/home/jinmyeong/code/hts/classifier/Data/groudTruth_hatemm_audioHateXplain.json"
                                , ASR_result_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/merged_whisperX-{whisper_type}_ASR_transcription_hatemm.json")
    print(f"whisper_{whisper_type}_WER")

# Whisper
whisper_size = ["tiny", "base", "small", "medium", "large", "large-v2"]
# for whisper_type in whisper_size:

#     whisper_hatemm_WER = get_WER_from_gold_ASR_transcript_whisper(gold_audioHateXplain_path="/home/jinmyeong/code/hts/classifier/Data/groudTruth_audioHateXplain.json"
#                                 , ASR_result_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/whisper-{whisper_type}_ASR_transcription_v4-HateSpeech-all-multispeaker.json")
#     print(f"whisper_{whisper_type}_WER")

# groundTruth ASR
whisper_hatemm_WER = get_WER_from_gold_ASR_transcript(gold_audioHateXplain_path="/home/jinmyeong/code/hts/classifier/Data/groudTruth_audioHateXplain.json"
                            ,ASR_result_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/groundTruth_ASR_transcription_audioHateXplain.json")

# audioHateXplain WhisperX
whisper_size = ["tiny", "base", "small", "medium", "large", "large-v2"]
for whisper_type in whisper_size:

    whisper_hatemm_WER = get_WER_from_gold_ASR_transcript(gold_audioHateXplain_path="/home/jinmyeong/code/hts/classifier/Data/groudTruth_audioHateXplain.json"
                                , ASR_result_path=f"/home/jinmyeong/code/hts/ASR/Data/ASR/merged_whisperX-{whisper_type}_ASR_transcription_audioHateXplain.json")
    print(f"whisper_{whisper_type}_WER")



# get_WER_from_gold_ASR_transcript(gold_audioHateXplain_path="/home/jinmyeong/code/hts/classifier/Data/audio_rationales_data.json"
#                                 , ASR_result_path='/home/jinmyeong/code/hts/ASR/Data/ASR/whisperX_ASR_transcription_v4-HateSpeech-all-multispeaker.json')