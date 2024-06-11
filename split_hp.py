import os, random, json, pickle

img_path = '/mnt/manhnd/DATA/stomach_hp_20220822/images/'
metadata_path = '/mnt/manhnd/DATA/stomach_hp_20220822/metadata/'

negative_hp = []
positive_hp = []

json_files = os.listdir(metadata_path)

for name in json_files:
    img_name = img_path + name.replace('.json', '.jpeg')
    with open(metadata_path + name) as f:
        tags = json.load(f)['image_tag_list']
        for tag in tags:
            if tag['display_name'] == 'HP dương tính':
                positive_hp.append(img_name)
            elif tag['display_name'] == 'HP âm tính':
                negative_hp.append(img_name)

random.shuffle(negative_hp)
random.shuffle(positive_hp)

res = [[], [], [], [], []]

neg_fold = int(len(negative_hp)/5)
pos_fold = int(len(positive_hp)/5)

for i in range(4):
    res[i] = [(n, 0) for n in negative_hp[i*neg_fold:(i+1)*neg_fold]] + [(n, 1) for n in positive_hp[i*pos_fold:(i+1)*pos_fold]]

res[4] = [(n, 0) for n in negative_hp[4*neg_fold:]] + [(n, 1) for n in positive_hp[4*pos_fold:]]

x = {
    0: res[0],
    1: res[1],
    2: res[2],
    3: res[3],
    4: res[4],
}

with open('hp_fold', 'wb') as f:
    pickle.dump(x, f)
