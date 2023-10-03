import cv2
import os
from typing import Union
from multiprocessing import Pool, cpu_count


def extract_frames_from_folder(
    folder_path: Union[str, os.PathLike],
    output_folder: Union[str, os.PathLike]
):
    """Extract frames from all videos in a folder"""
    for video in os.listdir(folder_path):
        video_path = os.path.join(folder_path, video)

        if os.path.isdir(video_path):
            continue

        video_output_path = os.path.join(output_folder, video.split(".")[0])
        total_extracted = extract_frames(video_path, video_output_path)
        print(f"Extracted {total_extracted} frames from {video_path}")


def process_chunk(args):
    """Process a chunk of frames"""
    video_path, output_folder, start_frame, end_frame, batch_size = args

    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    count = start_frame
    batch_frames = []

    while count <= end_frame:
        success, image = vidcap.read()
        if not success:
            break

        batch_frames.append(image)

        # When batch size is reached or end of chunk, save images
        if len(batch_frames) == batch_size or count == end_frame:
            for idx, frame in enumerate(batch_frames):
                cv2.imwrite(os.path.join(output_folder, f"frame{count + idx}.jpg"), frame)
            count += len(batch_frames)
            batch_frames.clear()

    print(f"Processed frames {start_frame} to {end_frame}")


def extract_frames(video_path, output_folder, batch_size=50):
    """Extract frames from a video using multiprocessing"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    cores = cpu_count()

    chunk_size = total_frames // cores
    chunks = [(i*chunk_size, (i+1)*chunk_size - 1) for i in range(cores)]
    chunks[-1] = (chunks[-1][0], total_frames - 1)  # Adjust the last chunk to include any remaining frames

    args = [(video_path, output_folder, start, end, batch_size) for start, end in chunks]
    
    with Pool(cores) as pool:
        pool.map(process_chunk, args)
