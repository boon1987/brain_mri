import os

import numpy as np
import pandas as pd

from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split

import shutil

import python_pachyderm
from python_pachyderm.pfs import Commit
from python_pachyderm.proto.v2.pfs.pfs_pb2 import FileType
import torch
from PIL import Image

from skimage import io
from torch.utils.data import Dataset



class MRI_Dataset(Dataset):
    def __init__(self, path_df, data_dir, transform=None):
        self.path_df = path_df
        self.transform = transform
        self.data_dir = data_dir
        
    def __len__(self):
        return self.path_df.shape[0]
    
    def __getitem__(self, idx):
        
        base_path = os.path.join(self.data_dir, self.path_df.iloc[idx]['directory'].strip("/"))
        img_path = os.path.join(base_path, self.path_df.iloc[idx]['images'])
        mask_path = os.path.join(base_path, self.path_df.iloc[idx]['masks'])
        
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        
        sample = (image, mask)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class PairedToTensor():
    def __call__(self, sample):
        img, mask = sample
        img = np.array(img)
        mask = np.expand_dims(mask, -1)
        img = np.moveaxis(img, -1, 0)
        mask = np.moveaxis(mask, -1, 0)
        img, mask = torch.FloatTensor(img), torch.FloatTensor(mask)
        img = img/255
        mask = mask/255
        return img, mask
    
def get_train_val_datasets(download_dir, data_dir, seed, validation_ratio=0.2):
    
    dirs, images, masks = [], [], []


    full_dir = "/"
    full_dir = os.path.join(full_dir, download_dir.strip("/"), data_dir.strip("/"))
    
    print("full_dir = " + full_dir)

    for root, folders, files in  os.walk(full_dir):
        for file in files:
            if 'mask' in file:
                dirs.append(root.replace(full_dir, ''))
                masks.append(file)
                images.append(file.replace("_mask", ""))

    
    PathDF = pd.DataFrame({'directory': dirs,
                          'images': images,
                          'masks': masks})

    
    train_df, valid_df = train_test_split(PathDF, random_state=seed,
                                     test_size = validation_ratio)
    

    
    train_data = MRI_Dataset(train_df, full_dir, transform=PairedToTensor())
    valid_data = MRI_Dataset(valid_df, full_dir, transform=PairedToTensor())
    
    return train_data, valid_data



# ======================================================================================================================
def safe_open_wb(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'wb')


def download_pach_repo(
    pachyderm_host,
    pachyderm_port,
    repo,
    branch,
    commit, 
    job_id, 
    root,
    token,
    project="default",
    previous_commit=None,
):
    print(f"Starting to download dataset from [{repo}@{branch}] to [{root}]")

    if not os.path.exists(root):
        os.makedirs(root)

    client = python_pachyderm.Client(host=pachyderm_host, port=pachyderm_port, auth_token=token)
    files = []
    if previous_commit is not None:
        print('download_pach_repo: previous_commit is not None.')
        print('previous_commit: ', previous_commit)
        for diff in client.diff_file(
            Commit(repo=repo, id=job_id, project=project), "/",     # retrieve the input commit of the current pachyderm pipeline job. repo is the repo_name containing the input dataset.
            Commit(repo=repo, id=previous_commit, project=project), # retrieve the input commit of the previous pachyderm pipeline job. Note that the "previous_commit" is the id of previous job, not the real repo id. repo is the repo_name containing the input dataset.
        ):
            src_path = diff.new_file.file.path
            des_path = os.path.join(root, src_path[1:])
            print(f"Got src='{src_path}', des='{des_path}'")

            if diff.new_file.file_type == FileType.FILE:
                if src_path != "":
                    files.append((src_path, des_path))
    else:
        print('download_pach_repo: previous_commit is None.')
        print(project, repo, branch, commit)
        commit_object = Commit(project=project, repo=repo, branch=branch, id=commit) # retrieve the specified commit to train the model. The "commit" here is the specific commit of the training dataset with the order of "project->repo->branch->commit".
        #commit_object = python_pachyderm.pfs.Commit(repo='mri_raw_data', branch='branch_v1', project='Brain-MRI', id='3c6021c926d145cfbef63c6a9519de57')
        for file_info in client.walk_file(commit_object, "/"):
            src_path = file_info.file.path
            des_path = os.path.join(root, src_path[1:])
            if file_info.file_type == FileType.FILE:
                if src_path != "":
                    files.append((src_path, des_path))
                    

    counter=0
    for src_path, des_path in files:
        src_file = client.get_file(Commit(repo=repo, id=branch, project=project), src_path)
        print(counter, ":", f"Downloading {src_path} to {des_path}")

        with safe_open_wb(des_path) as dest_file:
            shutil.copyfileobj(src_file, dest_file)
                
        # if counter==500:
        #     break
        counter=counter+1
        

    print("Download operation ended")
    return files

# ========================================================================================================
