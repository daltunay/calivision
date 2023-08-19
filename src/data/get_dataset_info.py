import json
import logging
import os
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


class DatasetInfoProcessor:
    def __init__(self, raw_folder: str, info_folder: str) -> None:
        self.raw_folder = raw_folder
        self.info_folder = info_folder

    def truncate_path(self, full_path: str) -> str:
        parts = full_path.split(os.path.sep)
        index = parts.index("dataset") - 1
        return os.path.sep.join(parts[index:])

    def process_dataset_folder(self, folder_name: str) -> Dict[str, List[str]]:
        folder_path = os.path.join(self.raw_folder, folder_name)
        dataset_content = {}
        for label in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label)
            if os.path.isdir(label_path):
                file_list = os.listdir(label_path)
                dataset_content[label] = file_list
        return dataset_content

    def save_json_file(self, data: Any, output_file: str) -> None:
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)

    def process_and_export(self) -> None:
        dataset_folders = [
            folder_name
            for folder_name in os.listdir(self.raw_folder)
            if os.path.isdir(os.path.join(self.raw_folder, folder_name))
        ]

        logging.info(f"Getting dataset info from: {self.truncate_path(self.raw_folder)}")

        for folder_name in dataset_folders:
            folder_path = os.path.join(self.raw_folder, folder_name)
            if os.path.isdir(folder_path):
                content = self.process_dataset_folder(folder_name)
                num_labels = len(content)
                num_videos = sum(len(videos) for videos in content.values())

                os.makedirs(self.info_folder, exist_ok=True)
                output_file = os.path.join(self.info_folder, f"{folder_name}.json")
                self.save_json_file(content, output_file)

                logging.info(
                    f"/{folder_name}/ dataset size: {num_labels} labels, {num_videos} videos ({num_videos / num_labels} videos/label). Info saved to {self.truncate_path(output_file)}."
                )


def main():
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_folder_path = os.path.join(parent_dir, "dataset", "raw")
    info_folder_path = os.path.join(parent_dir, "dataset", "info")

    processor = DatasetInfoProcessor(raw_folder_path, info_folder_path)
    processor.process_and_export()


if __name__ == "__main__":
    main()
