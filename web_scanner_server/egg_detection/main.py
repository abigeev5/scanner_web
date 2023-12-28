from inference import inference
from fiftyone_visualize import fiftyone_visualize

# папка для обработки
folder_path = 'egg_detection/test_files'

result_json_path = inference(folder_path)
fiftyone_visualize(folder_path, result_json_path)
