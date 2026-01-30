import random
import shutil
from pathlib import Path 
from tqdm import tqdm

## Paths to where the selected videos are stored and path where the split videos wil be stored
selected_videos = Path("system/data/selected")
split_path = Path("system/data/splitvids")

## Split videos into test and validation percent
training_split = 0.75
validate_split = 0.15

seed = 0

## Delete current split vidoes before copying new ones 
clear = True

## Function to load all the selected videos in
def load_videos(cls):
    return list((selected_videos / cls).glob("*.mp4"))

## Function to randomly determine how the videos are split into train, validation & test
def split_videos(videos, seed):
    rng = random.Random(seed)
    rng.shuffle(videos)
    
    num = len(videos)
    num_train = int(num * training_split)
    num_validate = int(num * validate_split)
    
    train = videos[:num_train]
    validate = videos[num_train:num_train + num_validate]
    test = videos[num_train + num_validate:]
    
    return train, validate, test

## Function to copy videos to the target directory
def copy_videos(videos, out_dir, desc):
    for src in tqdm(videos, desc=desc):
        shutil.copy2(src, out_dir / src.name)

## Function to split and copy the real and fake videos into the train, validate and test folders
def process_split(cls, seed):
    videos = load_videos(cls)
    train, validate, test = split_videos(videos, seed)
    
    print(f"{cls}: {len(train)} train / {len(validate)} validate / {len(test)} test")

    copy_videos(train, split_path / "train" / cls, f"Copying {cls} train")
    copy_videos(validate, split_path / "validate" / cls, f"Copying {cls} validate")
    copy_videos(test, split_path / "test" / cls, f"Copying {cls} test")


def main():
    process_split("real", seed + 1)
    process_split("fake", seed + 2)
    
    print("Splitting complete! Videos are in: ", split_path)
    
if __name__ == "__main__":
    main()
    
    