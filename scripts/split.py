''' split.py
Split Data into train (80%) and validate (20%)
Run from root of project as
python scripts/split.py
'''

import os
import glob
import shutil

def mkdir(dirname):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)

VALIDATE_DIR = "data/validate"
mkdir(VALIDATE_DIR)

TRAIN_DIR = "data/train"
mkdir(TRAIN_DIR)
DOG_FILENAMES = glob.glob("data/raw/dog*")
CAT_FILENAMES = glob.glob("data/raw/cat*")

# N_DOG_TEST = int(len(DOG_FILENAMES) * .2)
# N_CAT_TEST = int(len(CAT_FILENAMES) * .2)

# Move 20% to validate directory

DOG_TRAIN = "data/train/dogs"
mkdir(DOG_TRAIN)
DOG_VALIDATE = "data/validate/dogs"
mkdir(DOG_VALIDATE)

for f in DOG_FILENAMES:
    _, index, _ = f.split(".")
    if 0 <= int(index) < 1000:
        shutil.copy(f, DOG_TRAIN)
    elif 1000 <= int(index) < 1400:
        shutil.copy(f, DOG_VALIDATE)

        
CAT_TRAIN = "data/train/cats"
mkdir(CAT_TRAIN)
CAT_VALIDATE = "data/validate/cats"
mkdir(CAT_VALIDATE)
for f in CAT_FILENAMES:
    _, index, _ = f.split(".")
    if 0 <= int(index) < 1000:
        shutil.copy(f, CAT_TRAIN)
    elif 1000 <= int(index) < 1400:
        shutil.copy(f, CAT_VALIDATE)

print "{0}: {1} files, {2}: {3} files".format(DOG_TRAIN, len(os.listdir(DOG_TRAIN)),
    DOG_VALIDATE, len(os.listdir(DOG_VALIDATE)))
print "{0}: {1} files, {2}: {3} files".format(CAT_TRAIN, len(os.listdir(CAT_TRAIN)),
    CAT_VALIDATE, len(os.listdir(CAT_VALIDATE)))
