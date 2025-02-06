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
ROOT_DIR = r"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\mini_dataset\train"
video_files = os.listdir(os.path.join(ROOT_DIR, "images"))

# %%
for video_file in video_files:

    print(video_file)
    images_dir = os.path.join(ROOT_DIR, "images", video_file)
    masks_dir = os.path.join(ROOT_DIR, "annotations", video_file)
    output_dir = rf"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\validate\{video_file}"

    overlay_masks(images_dir, masks_dir, output_dir)

# %%
print("Finito")

# %%
