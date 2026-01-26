import random
import shutil
from pathlib import Path 
from tqdm import tqdm

## Paths to the raw dataset on the local Dev Machine (Darragh's Machine)
real_path = Path(r"C:\Users\darra\Downloads\DFD Dataset\DFD_original sequences")
fake_path = Path(r"C:\Users\darra\Downloads\DFD Dataset\DFD_manipulated_sequences\DFD_manipulated_sequences")

output_real = Path("system/data/selected/real")
output_fake = Path("system/data/selected/fake")

## Matching the number of fakes with the number of real videos in the dataset
num_fakes = 364
seed = 0

## Function to return all the videos that are of MP4 format / Raise error if no the paths don't exist
def find_videos(root: Path):
    if not root.exists():
        raise FileNotFoundError(root)
    return list(root.rglob("*.mp4"))

## Function that shuffles what videos are to be selected
def random_shuffle(videos, n, seed):
    rng = random.Random(seed)
    videos = list(videos)
    rng.shuffle(videos)
    step = len(videos) / n      ## Step creates how many chunks are in the set, chunk 364 would consist of the videos at the end of the dataset
    return [rng.choice(videos[int(i*step):int((i+1)*step)] or videos) for i in range(n)]

## The main function that will copy the videos slected and paste them into their respective real or fake folders
def main():
    if not output_real.exists() or not output_fake.exists():
        raise FileNotFoundError()

    real_videos = list(real_path.rglob("*.mp4"))
    fake_videos = list(fake_path.rglob("*.mp4"))
    
    ## Conform number of rael and fakes videos found
    print("Number of real videos:", len(real_videos))
    print("Number of fake videos:", len(fake_videos))
    
    ## Confirm number of fake videos selected
    selected_fakes = random_shuffle(fake_videos, num_fakes, seed)
    print("Number of fakes selected:", len(selected_fakes))
    
    ## Copy all real videos to the real folder (364 videos)
    for i, src in enumerate(tqdm(real_videos, desc = "Coping Real Videos"), 1):
        shutil.copy2(src, output_real / f"real_{i:04d}.mp4")        ## Rename each video file with the same ID format
        
    ## Copy the 364 selected fake videos to the fake folder
    for i, src in enumerate(tqdm(selected_fakes, desc = "Coping Fake Videos"), 1):
        shutil.copy2(src, output_fake / f"fake_{i:04d}.mp4")
        
    
    print("Video Selection and Copying Complete")

if __name__ == "__main__":
    main()