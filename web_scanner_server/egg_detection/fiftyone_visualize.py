import fiftyone as fo
import fiftyone.zoo as foz


def fiftyone_visualize(img_dir, result_json_path):
    # The directory containing the source images
    data_path = img_dir

    # The path to the COCO labels JSON file
    labels_path = result_json_path


    # Import the dataset
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
        label_field="FasterRCNN"
    )

    session = fo.launch_app(dataset)

    session.wait()
