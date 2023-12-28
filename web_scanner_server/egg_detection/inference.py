import json
import os
from PIL import Image
from datetime import datetime

from . import inference_dependencies


def generate_empty_json_only_filenames(img_dir):
    base_folder_name = os.path.basename(img_dir)
    dataset_name = base_folder_name

    classes = ('Enterobius vermicularis',)

    # list all images in directory
    img_paths = [[f.path, f.name] for f in sorted(os.scandir(img_dir), key=lambda e: e.name)
                 if f.name.split('.')[-1] in ('jpg', 'png', 'jpeg', 'bmp')]
    images = []
    id = 1
    for img_path, img_name in img_paths:
        img = Image.open(img_path)
        width, height = img.size
        images.append({'id': id,
                       'file_name': img_name,
                       'height': height,
                       'width': width,
                       'license': None,
                       'coco_url': None})
        id += 1

    # add some general info
    info = {'date': datetime.today().strftime('%Y-%m-%d'),
            'author': 'kvs',
            'description': dataset_name}

    categories = []
    categories.append({'id': 0,
                           'name': 'Enterobius vermicularis'})
    # categories.append({'id': 0,
    #                        'name': ''})

    coco_json = {'info': info,
                 'licenses': [],
                 'categories': categories,
                 'images': images,
                 'annotations': []}

    # save output json file
    empty_json_path_only_filenames = base_folder_name + '_empty_json_only_filenames.json'
    with open(empty_json_path_only_filenames, 'w') as file:
        json.dump(coco_json, file)

    return empty_json_path_only_filenames


def inference(img_dir):
    empty_json_path_only_filenames = generate_empty_json_only_filenames(img_dir)

    results = inference_dependencies.predict_my_version(
        model_type="mmdet",
        model_path='egg_detection/model_fasterrcnn_23_11.pth',
        model_config_path='egg_detection/config_fasterrcnn_23_11.py',
        model_device='cuda:0',
        model_confidence_threshold=0.975,
        source=img_dir,
        slice_height=600,
        slice_width=800,
        overlap_height_ratio=0.15,
        overlap_width_ratio=0.15,
        novisual=True,
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.8,
        dataset_json_path=empty_json_path_only_filenames
    )

    with open(empty_json_path_only_filenames) as f:
        json_empty = json.load(f)

    with open('result.json') as f:
        json_result = json.load(f)

    json_empty['annotations'] = json_result

    base_folder_name = os.path.basename(img_dir)
    json_filename_full_result = base_folder_name + '_full_result.json'

    with open(json_filename_full_result, 'w') as g:
        json.dump(json_empty, g)

    return json_filename_full_result
