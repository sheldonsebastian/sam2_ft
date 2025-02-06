# %%
from sklearn.model_selection import train_test_split
import os
from PIL import Image

# %%
# List all files in the prepped_mini_dataset folder
file_list = os.listdir(os.path.join("prepped_mini_dataset", "images"))

# %%
len(file_list)

# %%
# Split the dataset into train and test sets
train, test = train_test_split(file_list, test_size=0.2, random_state=42)

# %%
len(train)

# %%
len(test)

# %%
# create folder for train and test
os.makedirs(os.path.join("mini_dataset", "train", "annotations"), exist_ok=True)
os.makedirs(os.path.join("mini_dataset", "train", "images"), exist_ok=True)
os.makedirs(os.path.join("mini_dataset", "test", "annotations"), exist_ok=True)
os.makedirs(os.path.join("mini_dataset", "test", "images"), exist_ok=True)


# %%
def resize_and_save(src_path, dest_path, size=(1024, 1024)):
    os.makedirs(dest_path, exist_ok=True)
    for file_name in sorted(os.listdir(src_path)):
        with Image.open(os.path.join(src_path, file_name)) as img:
            img = img.resize(size)
            img.save(os.path.join(dest_path, file_name))


# %%
# copy train folders
for file in train:
    resize_and_save(
        os.path.join(
            r"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\prepped_mini_dataset",
            "images",
            file,
        ),
        os.path.join(
            r"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\mini_dataset",
            "train",
            "images",
            file,
        ),
    )
    resize_and_save(
        os.path.join(
            r"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\prepped_mini_dataset",
            "annotations",
            file,
        ),
        os.path.join(
            r"C:\Users\sheld\OneDrive\Workspaces\sam2_ft\mini_dataset",
            "train",
            "annotations",
            file,
        ),
    )

# %%
# copy test folders using shutil
for file in test:
    resize_and_save(
        os.path.join("prepped_mini_dataset", "images", file),
        os.path.join("mini_dataset", "test", "images", file),
    )
    resize_and_save(
        os.path.join("prepped_mini_dataset", "annotations", file),
        os.path.join("mini_dataset", "test", "annotations", file),
    )

# %%
print("Finito")

# %%
