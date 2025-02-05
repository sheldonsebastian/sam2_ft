# %%
import cv2
import os
import json
import numpy as np
from pycocotools import mask as maskUtils


# %%
def extract_frames(video_path, output_folder, frame_interval=4):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def rle_to_mask(rle, height, width):
    """
    Convert RLE to binary mask.

    :param rle: Run-length encoding
    :param height: Height of the mask
    :param width: Width of the mask
    :return: Binary mask
    """
    rle["size"] = [height, width]
    binary_mask = maskUtils.decode(rle)
    return binary_mask


def save_mask_as_png(mask, output_path):
    """
    Save binary mask as PNG image.

    :param mask: Binary mask
    :param output_path: Path to save the PNG image
    """
    cv2.imwrite(output_path, mask)


def create_combined_masks(data, output_folder):
    combined_masks = {}

    for frame_num, masklets_arr in enumerate(data["masklet"]):
        for obj_id, mask_data in enumerate(masklets_arr):
            height, width = mask_data["size"]
            rle = {"counts": mask_data["counts"], "size": [height, width]}
            binary_mask = rle_to_mask(rle, height, width)

            # Initialize the combined mask for the frame if not already done
            if frame_num not in combined_masks:
                combined_masks[frame_num] = np.zeros((height, width, 3), dtype=np.uint8)

            # Assign a unique color to each object in the mask
            color = (obj_id * 50 % 256, obj_id * 100 % 256, obj_id * 150 % 256)
            combined_masks[frame_num][binary_mask > 0] = color

    # Save the combined masks for each frame
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for frame_num, combined_mask in combined_masks.items():
        output_path = os.path.join(output_folder, f"{str(frame_num).zfill(5)}.png")
        save_mask_as_png(combined_mask, output_path)


# %% EXTRACT FRAMES
video_files = [
    "sav_000001.mp4",
    "sav_000002.mp4",
    "sav_000003.mp4",
    "sav_000004.mp4",
]

for i, video_file in enumerate(video_files, start=1):
    extract_frames(
        os.path.join(
            r"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\raw_mini-dataset", video_file
        ),
        os.path.join(
            r"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\prepped_mini_dataset",
            "images",
            f"video_{i:04d}",
        ),
        frame_interval=4,
    )

# %% CREATE MASKS
video_mask_json_files = [
    "sav_000001_auto.json",
    "sav_000002_auto.json",
    "sav_000003_auto.json",
    "sav_000004_auto.json",
]

for i, video_mask_json_path in enumerate(video_mask_json_files, start=1):
    # Load the JSON file
    with open(
        os.path.join(
            r"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\raw_mini-dataset",
            video_mask_json_path,
        ),
    ) as f:
        data = json.load(f)

    output_folder = os.path.join(
        r"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\prepped_mini_dataset",
        "annotations",
        f"video_{i:04d}",
    )
    create_combined_masks(data, output_folder)


# %%
print("Finito")

# %%
