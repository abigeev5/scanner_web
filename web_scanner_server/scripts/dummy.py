import os
import tqdm
import json


output = {'results': {}}
for folder in tqdm.tqdm(os.listdir('results')):
    output['results'][folder] = list()
    for path in os.listdir(f'results/{folder}'):
        if ('.png' in path):
            output['results'][folder].append(
                {'image': path, 
                 'bboxes': [
                     [200, 200, 300, 400], 
                     [100, 100, 300, 400], 
                     [300, 300, 500, 500]], 
                 'time': '10.10.22 15:15:15', 
                 'thumbnail': f'thumb_{path.replace(".png", ".jpg")}'})

with open('dummy.json', 'w') as f:
    json.dump({'results': output['results']['1509698406']}, f)