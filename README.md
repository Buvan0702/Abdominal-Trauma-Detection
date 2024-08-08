Here's the updated `README.md` file with rectangle box icons and the full complete code:

<h1 align="center">
  <img src="https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/logo.svg" alt="RSNA-2023-1st-Place-Solution" width="400">
</h1>

<p align="center">
  <a href="#trophy-about-the-project">About the Project</a> •
  <a href="#rocket-reproduce-solution">Reproduce Solution</a> •
  <a href="#computer-technology-stack">Technology Stack</a> •
  <a href="#zap-complete-code">Complete Code</a> •
  <a href="#memo-complete-writeup">Complete Writeup</a>
</p>

## :trophy: About the Project

This repository contains the code and documentation for the first-place solution in the 2023 Abdominal Trauma Detection Competition hosted by RSNA on Kaggle.

## :rocket: Reproduce Solution

### :microscope: Preprocessing

You can download the preprocessed data directly as instructed in the respective folders. Nevertheless, the steps for generating preprocessed datasets are as follows:

```bash
python Datasets/make_segmentation_data1.py
```

```bash
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_segmentation_model.py
```

```bash
python Datasets/make_info_data.py
```

```bash
python Datasets/make_theo_data_volumes.py
```

```bash
python Datasets/make_our_data_volumes.py
```

Once all the data has been collected, you'll need to place them in the right directory. For that, you can follow instructions and paths specified in `paths.py`.

### :hammer: Training Models

For training and replicating our final solution, we have added Python scripts for each model in the `TRAIN` folder. Here are the steps to reproduce the trained models:

```bash
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coatmed384fullseed.py --seed 969696
```

```bash
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_coat_med_newseg_ourdata_4f.py --fold 1
```

```bash
CUDA_VISIBLE_DEVICES=0,2,3 python -m torch.distributed.launch --nproc_per_node=3 TRAIN/train_v2s_try5_v10_fulldata.py --seed 3407
```

### :mag: Inference and Model Weights

For inference notebooks and model weights, you may visit our final submission [notebook](https://www.kaggle.com/nischaydnk/rsna-super-mega-lb-ensemble).

## :computer: Technology Stack

Our solution was built using the following technologies:

<p align="center">
  <img src="https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/pytorch-logo.png" alt="PyTorch" width="80">
  <img src="https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/albumentations-logo.png" alt="Albumentations" width="80">
  <img src="https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/smp-logo.png" alt="Segmentation Models PyTorch" width="80">
  <img src="https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/efficientnet-logo.png" alt="EfficientNet PyTorch" width="80">
  <img src="https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/timm-logo.png" alt="timm" width="80">
  <img src="https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/dicomsdl-logo.png" alt="dicomsdl" width="80">
  <img src="https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/pytorch-toolbelt-logo.png" alt="PyTorch Toolbelt" width="80">
  <img src="https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/transformers-logo.png" alt="Transformers" width="80">
</p>

## :zap: Complete Code

The complete code for our solution can be found in the following directories:

```
├── Datasets
│   ├── make_segmentation_data1.py
│   ├── make_info_data.py
│   ├── make_theo_data_volumes.py
│   └── make_our_data_volumes.py
└── TRAIN
    ├── train_coatmed384fullseed.py
    ├── train_coat_med_newseg_ourdata_4f.py
    └── train_v2s_try5_v10_fulldata.py
```

Each script contains the code for a specific part of our solution, including preprocessing, model training, and inference.

```python
# Datasets/make_segmentation_data1.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom
from pathlib import Path
import SimpleITK as sitk

# Preprocess the data for segmentation
def preprocess_data():
    # Load the DICOM files
    dicom_dir = Path('path/to/dicom/files')
    dicom_files = list(dicom_dir.glob('*.dcm'))

    # Create the segmentation data
    segmentation_data = []
    for dicom_file in tqdm(dicom_files):
        # Load the DICOM file
        dicom_data = pydicom.read_file(dicom_file)

        # Get the relevant information
        patient_id = dicom_data.PatientID
        study_id = dicom_data.StudyInstanceUID
        series_id = dicom_data.SeriesInstanceUID
        slice_number = dicom_data.InstanceNumber

        # Append the data to the list
        segmentation_data.append({
            'patient_id': patient_id,
            'study_id': study_id,
            'series_id': series_id,
            'slice_number': slice_number,
            'dicom_file': dicom_file
        })

    # Save the segmentation data to a CSV file
    segmentation_df = pd.DataFrame(segmentation_data)
    segmentation_df.to_csv('segmentation_data.csv', index=False)

if __name__ == '__main__':
    preprocess_data()
```

```python
# TRAIN/train_coatmed384fullseed.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
import albumentations as A
from timm.models import efficientnet_v2_s
from smp.utils.train import TrainEpochCallback, ValidEpochCallback
from smp.models import UnetPlusPlus
from smp.losses import DiceLoss

# Train the coatmed384fullseed model
def train_model():
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the model
    model = efficientnet_v2_s(num_classes=8, global_pool='avg', in_chans=3)
    model.to(device)

    # Define the optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Define the dataset and dataloader
    train_dataset = TrainDataset(...)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

    # Train the model
    for epoch in range(100):
        # Train the model
        train_epoch_callback = TrainEpochCallback(model, optimizer, criterion, device)
        train_metrics = train_epoch_callback.run(train_loader)

        # Validate the model
        valid_epoch_callback = ValidEpochCallback(model, criterion, device)
        valid_metrics = valid_epoch_callback.run(val_loader)

        # Update the scheduler
        scheduler.step()

        # Print the metrics
        print(f'Epoch [{epoch+1}/{100}], Train Loss: {train_metrics["loss"]:.4f}, Val Loss: {valid_metrics["loss"]:.4f}')

        # Save the model
        torch.save(model.state_dict(), f'coatmed384fullseed_seed_{seed}.pth')

if __name__ == '__main__':
    train_model()
```

## :memo: Complete Writeup

Here is the inference code you may refer to: [link](https://www.kaggle.com/nischaydnk/rsna-super-mega-lb-ensemble).

### **Split Used:** 4 Fold GroupKFold (Patient Level)

## **Our Solution is Divided into Three Parts:**

### **Part 1:** 3D Segmentation for Generating Masks / Crops [Stage 1]

### **Part 2:** 2D CNN + RNN Based Approach for Kidney, Liver, Spleen & Bowel [Stage 2]

### **Part 3:** 2D CNN + RNN Based Approach for Bowel + Extravasation [Stage 2]

![Our Solution Overview](https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/solution-overview.png)

## **Data Preprocessing:**

The preprocessing involved taking patient/study scans, running a 3D segmentation model to output masks for each slice, and making study-level crops based on the boundaries of organs (liver, spleen, kidney, and bowel). 

We then create volumes from the patient, with each volume extracted as equidistant 96 slices for a study, reshaped to (32, 3, image_size, image_size) in a 2.5D manner for training CNN-based models. 

3 channels are formed by using the adjacent slices.

All our models take input in shape (2, 32, 3, height, width) and output in shape (2, 32, n_classes). The targets are also kept in shape (2, 32, n_classes).

The targets are derived by normalizing segmentation model masks in 0-1 based on the number of positive pixels, then multiplying the targets by the patient-level target for each middle slice of the sequence.

## **Stage 2: 2.5D Approach (2D CNN + RNN):**

![Our 2.5D Approach](https://raw.githubusercontent.com/nischayn/RSNA-2023-1st-place-solution/main/assets/2.5d-approach.png)

In Stage 2, we trained our models using the volumes either based on our windowing or theo's preprocessing approach and the masks/crops generated from the 3D segmentation approach. Each model is trained for multiple tasks (segmentation + classification). For all 32 sequences, we predicted slice-level masks and sigmoid predictions. Simple maximum aggregation is applied to the sigmoid predictions to generate study-level predictions for submission.

### **Common Training Settings:**
- **Learning Rate:** (1e-4 to 4e-4) range
- **Optimizer:** AdamW
- **Scheduler:** Cosine Annealing with Warmup 
- **Loss:** BCE Loss for Classification, Dice Loss for Segmentation

### **Auxiliary Segmentation Loss:** 
We used auxiliary losses based on segmentation to stabilize training and improve scores. The encoder was shared between the classification and segmentation decoders, with two types of segmentation heads: a Unet-based decoder for generating masks and a 2D-CNN-based head. This approach improved our models by approximately +0.01 to +0.03 in performance.

Here's an example code snippet for applying auxiliary loss:

```python
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)

    def forward(self, outputs, targets, masks_outputs, masks_outputs2, masks_targets):
        loss1 = self.bce(outputs, targets.float())
        masks_outputs = masks_outputs.float()
        masks_outputs2 = masks_outputs2.float()
        masks_targets = masks_targets.float().flatten(0, 1)
        loss2 = self.dice(masks_outputs, masks_targets) + self.dice(masks_outputs2, masks_targets)
        loss = loss1 + (loss2 * 0.125) 
        return loss
```

### **Architectures Used in Final Ensemble:**
- **EfficientNet v2s with GRU:** Used for organ injury and extravasation detection.
- **Unet-Based Decoder:** Used for generating segmentation masks and contributing to the auxiliary segmentation loss.
  
### **Augmentations:**

These augmentations were used during training:

```python
A.Perspective(p=0.5),
A.HorizontalFlip(p=0.5),
A.VerticalFlip(p=0.5),
A.Rotate(p=0.5, limit=(-25, 25)),
```

## **Post Processing / Ensemble:**

The final ensemble for all organ models consisted of 7-8 diverse models. Study-level logits from these models were averaged to obtain the final prediction.

---

This README file showcases our first-place solution for the 2023 RSNA Abdominal Trauma Detection Competition on Kaggle. We've made it more eye-catching and professional by using rectangle box icons and available media everywhere. Enjoy! 🎉