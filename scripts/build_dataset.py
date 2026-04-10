from src.data.collect_raw_data import main as collect_webcam_data
from src.pipelines.run_dataset import run as run_dataset

if __name__ == "__main__":
    # collect_webcam = input("Do want to collect webcam data? (y/n): ")
    # if collect_webcam.lower() == "y":
    #     collect_webcam_data()

    # build_dataset = input("Do you want to build the dataset? (y/n): ")
    # if build_dataset.lower() == "y":
        run_dataset()