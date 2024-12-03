import json

gold_hatemm_hate_audioHateXplain_path = "/home/jinmyeong/code/hts/classifier/Data/gold_hatemm_audioHateXplain_hate+normal.json"
with open(gold_hatemm_hate_audioHateXplain_path, "r") as f: # from "Data" folder
    gold_hatemm_hate_audioHateXplain = json.load(f)


gold_hatemm_normal_audioHateXplain_path = "/home/jinmyeong/code/hts/classifier/Data/hatemm/hatemm_normal_human_transcript.json"
with open(gold_hatemm_normal_audioHateXplain_path, "r") as f: # from "Data" folder
    gold_hatemm_normal = json.load(f)

for doc_id, word_list in gold_hatemm_normal.items():
    sentence = " ".join(word_list)
    len_sentence = len(word_list)

    gold_hatemm_hate_audioHateXplain[doc_id]['sentence'] = sentence
    gold_hatemm_hate_audioHateXplain[doc_id]['rationale'] = [[0] * len_sentence, [0] * len_sentence, [0] * len_sentence]

for doc_id, value in gold_hatemm_hate_audioHateXplain.items():
    value['sentence'] = value['sentence'].split(" ")



file_path = '/home/jinmyeong/code/hts/classifier/Data/groudTruth_hatemm_audioHateXplain.json'
# with open(file_path, "w") as f:
#     json.dump(gold_hatemm_hate_audioHateXplain, f) 


gold_audioHateXplain_path = "/home/jinmyeong/code/hts/classifier/Data/audio_rationales_data.json"
with open(gold_audioHateXplain_path, "r") as f: # from "Data" folder
    gold_audioHateXplain = json.load(f)
# test ------
# c = 0
# for doc_id, value in gold_audioHateXplain.items():
#     if value['label'] != 'normal' and len(value['sentence']) != len(value['rationale'][0]):
#         c += 1
#         print(value)
# print(c)
# ------ test
for doc_id, value in gold_audioHateXplain.items():
    len_sentence = len(value['sentence'])
    if value['label'] == 'normal': 
        gold_audioHateXplain[doc_id]['rationale'] = [[0] * len_sentence, [0] * len_sentence, [0] * len_sentence]

for doc_id, value in gold_audioHateXplain.items():
    if doc_id == '14441243_gab':
        gold_audioHateXplain[doc_id]['intervals']['intervals [25]'] = {'xmin': 7.13, 'xmax': 7.19, 'text': 'a'}
        gold_audioHateXplain[doc_id]['intervals']['intervals [26]'] = {'xmin': 7.19, 'xmax': 7.5, 'text': 'jew'}
    if doc_id == '18779343_gab':
        gold_audioHateXplain[doc_id]['sentence'] = gold_audioHateXplain[doc_id]['sentence'][:19]

        del gold_audioHateXplain[doc_id]['intervals']['intervals [20]']
        del gold_audioHateXplain[doc_id]['intervals']['intervals [21]']
        del gold_audioHateXplain[doc_id]['intervals']['intervals [22]']
        for i, _ in enumerate(gold_audioHateXplain[doc_id]['rationale']):
            gold_audioHateXplain[doc_id]['rationale'][i] = gold_audioHateXplain[doc_id]['rationale'][i][:19]
        print()
    # TODO: word 길이가 rationale과 다른 경우, word 길이에 맞춰 rationale, interval 길이 조정
    sentence = " ".join(value['sentence'])
    word_list = sentence.split(" ")
    if len(value['rationale'][0]) != len(word_list):
        prev_rationale, prev_intervals = value["rationale"], value["intervals"]
        new_rationale, new_intervals_list = [[] for _ in range(len(prev_rationale))], []
        new_intervals = {}

        for word_idx, word in enumerate(value['sentence']): # 기존 데이터의 word가 1개 word로 이뤄졌는지 check
            len_word  = len(word.split(" "))
            word_segment_list = word.split(" ")
            if len_word != 1:
                for rationale_idx, _ in enumerate(new_rationale):
                    new_rationale[rationale_idx] += [prev_rationale[rationale_idx][word_idx]] * len_word
                prev_intervals_values = prev_intervals[f"intervals [{word_idx + 1}]"]
                
                xmax, xmin = prev_intervals_values['xmax'], prev_intervals_values["xmin"]
                xunit = round((xmax - xmin) / len_word, 2)

                for i in range(len_word):
                    if i == 0:
                        new_intervals_list.append({'xmin': xmin,'xmax': round(xmin+(i+1)*xunit, 2), 'text': word_segment_list[i]})
                    elif i == len_word - 1:
                        new_intervals_list.append({'xmin': round(xmin+i*xunit, 2),'xmax': xmax, 'text': word_segment_list[i]})
                    else:
                        new_intervals_list.append({'xmin': round(xmin+i*xunit, 2),'xmax': round(xmin+(i+1)*xunit, 2), 'text': word_segment_list[i]})
            else:
                for rationale_idx, _ in enumerate(new_rationale):
                    new_rationale[rationale_idx] += [prev_rationale[rationale_idx][word_idx]]
                prev_intervals_value = prev_intervals[f"intervals [{word_idx + 1}]"]
                new_intervals_list.append(prev_intervals_value)
        for idx, new_intervals_dict in enumerate(new_intervals_list):
            new_intervals[f"intervals [{idx + 1}]"] = new_intervals_dict
        gold_audioHateXplain[doc_id]['sentence'] = word_list
        gold_audioHateXplain[doc_id]['intervals'] = new_intervals
        gold_audioHateXplain[doc_id]['rationale'] = new_rationale

file_path = '/home/jinmyeong/code/hts/classifier/Data/groudTruth_audioHateXplain.json'
with open(file_path, "w") as f:
    json.dump(gold_audioHateXplain, f) 

print()
