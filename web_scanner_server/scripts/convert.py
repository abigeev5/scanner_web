import cv2
import os
import tqdm


for folder in tqdm.tqdm(os.listdir('results')):
    for path in os.listdir(f'results/{folder}'):
        img = cv2.imread(f"results/{folder}/{path}")
        if ('.bmp' in path):
            os.remove(f"results/{folder}/{path}")
            cv2.imwrite(f"results/{folder}/{path.split('.')[0]}.png", img, [cv2.IMWRITE_PNG_COMPRESSION, 100])
            thumb = cv2.resize(img, (160, 120))
            cv2.imwrite(f"results/{folder}/thumb_{path.split('.')[0]}.jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])
        elif ('.png' in path):
            thumb = cv2.resize(img, (160, 120))
            cv2.imwrite(f"results/{folder}/thumb_{path.split('.')[0]}.jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 70])