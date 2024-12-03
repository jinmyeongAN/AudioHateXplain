import json
import os
import torchaudio
from tqdm import tqdm

with open('/home/jinmyeong/code/hts/classifier/Data/hatemm_post_id_divisions.json', 'r') as fp:
    post_id_division = json.load(fp)

train_id_list, val_id_list, test_id_list = post_id_division['train'], post_id_division['val'], post_id_division['test']
len_train, len_val, len_test = len(train_id_list), len(val_id_list), len(test_id_list)

audio_filename_list = os.listdir("/mnt/hdd4/jinmyeong/hts_data/sliced_audio")
filepath = "/mnt/hdd4/jinmyeong/hts_data/sliced_audio"

# measure the audio time length
audio_filepath_list = [os.path.join(filepath, filename) for filename in audio_filename_list]

def get_average_lastTime(audio_id_list):
    last_time_list = []

    for key in tqdm(audio_id_list):
        for idx_audio, audio_filename in enumerate(audio_filename_list):
            if audio_filename[:-4] == key:
                info = torchaudio.info(audio_filepath_list[idx_audio])
                last_time = info.num_frames / info.sample_rate

                last_time_list.append(last_time)
    average_last_time = sum(last_time_list) / len(last_time_list) if len(last_time_list) != 0 else 0

    return average_last_time

train_average_lastTime = get_average_lastTime(train_id_list)
val_average_lastTime = get_average_lastTime(val_id_list)
test_average_lastTime = get_average_lastTime(test_id_list)

print(f"len_train: {len_train}")
print(f"len_val: {len_val}")
print(f"len_test: {len_test}")

print(f"train_average_lastTime: {train_average_lastTime}")
print(f"val_average_lastTime: {val_average_lastTime}")
print(f"test_average_lastTime: {test_average_lastTime}")

print()