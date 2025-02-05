# %%
import os
from PIL import Image


# %%
def overlay_masks(images_dir, masks_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(images_dir):
        if image_name.endswith(".jpg"):
            image_path = os.path.join(images_dir, image_name)
            mask_name = image_name.replace(".jpg", ".png")
            mask_path = os.path.join(masks_dir, mask_name)

            if os.path.exists(mask_path):
                image = Image.open(image_path).convert("RGBA")
                mask = Image.open(mask_path).convert("RGBA")

                overlay = Image.blend(image, mask, 0.5)

                # Save the overlayed image
                output_path = os.path.join(output_dir, mask_name)
                overlay.save(output_path)


# %%
video_files = [
    "video_0001",
    "video_0002",
    "video_0003",
    "video_0004",
]

for video_file in video_files:

    images_dir = rf"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\prepped_mini_dataset\images\{video_file}"
    masks_dir = rf"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\prepped_mini_dataset\annotations\{video_file}"
    output_dir = rf"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\validate\{video_file}"

    overlay_masks(images_dir, masks_dir, output_dir)

# %%
print("Finito")

# %%
