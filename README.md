Here’s the modified `README.md` file that only includes the information about the 3D segmentation model, 2D CNN + RNN, Unet, and EfficientNet models:

---

# RSNA-2023-1st-place-solution
Code and Documentation for the first place solution in the 2023 Abdominal Trauma Detection Competition hosted by RSNA on Kaggle.

## Reproduce Solution 

### Preprocessing
You can download the preprocessed theo data directly as instructed in the respective folders. Nevertheless, the steps for generating preprocessed datasets are as follows:

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

Once all the data has been collected, you will need to place them in the right directory. For that, you can follow instructions and paths specified in `paths.py`.

### Training Models 

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

### Inference and Model Weights

For inference notebooks and model weights, you may visit our final submission [notebook](https://www.kaggle.com/nischaydnk/rsna-super-mega-lb-ensemble).

## Hardware

All of our models were trained using multiple 3090 (local) or 3X A6000 instances with GPU enabled to run all data preprocessing, model training, and inference on Kaggle notebooks.

[https://www.kaggle.com/docs/notebooks](https://www.kaggle.com/docs/notebooks)

## Software

We used [Kaggle GPU notebooks](https://github.com/Kaggle/docker-python/blob/master/gpu.Dockerfile) to run all our inference scripts.

Below are the packages used in addition to the ones included in the default train scripts provided. All packages were installed via uploaded Kaggle dataset.

| Package Name                  | Repository                                                     | Kaggle Dataset                                          |
| ----------------------------- | -------------------------------------------------------------- | ------------------------------------------------------- |
| pytorch 2.0.1                  | https://github.com/pytorch/pytorch                             |                                                         |
| pretrained models 0.7.4        | https://pypi.org/project/pretrainedmodels/                     | https://www.kaggle.com/datasets/harshitsheoran/contrails-libraries |
| segmentation_models_pytorch 0.3.3 | https://github.com/qubvel/segmentation_models.pytorch       | https://www.kaggle.com/datasets/harshitsheoran/contrails-libraries |
| efficientnet_pytorch 0.7.1     | https://github.com/lukemelas/EfficientNet-PyTorch              | https://www.kaggle.com/datasets/harshitsheoran/contrails-libraries |
| albumentations                 | https://github.com/albumentations-team/albumentations          |                                                         |
| timm 0.9.7                     | https://github.com/rwightman/pytorch-image-models              |                                                         |
| dicomsdl 0.109.2               | https://github.com/tsangel/dicomsdl                            | https://www.kaggle.com/datasets/zacstewart/dicomsdl--0-109-2 |
| pytorch-toolbelt               | https://github.com/BloodAxe/pytorch-toolbelt                   | https://www.kaggle.com/bloodaxe/pytorch-toolbelt        |
| transformers 4.31.0            | https://github.com/huggingface/transformers                    |                                                         |

## Complete Writeup

Here is the inference code you may refer to: [link](https://www.kaggle.com/nischaydnk/rsna-super-mega-lb-ensemble).

### **Split Used:** 4 Fold GroupKFold (Patient Level)

## **Our Solution is Divided into Three Parts:**

### **Part 1:** 3D Segmentation for Generating Masks / Crops [Stage 1]

### **Part 2:** 2D CNN + RNN Based Approach for Kidney, Liver, Spleen & Bowel [Stage 2]

### **Part 3:** 2D CNN + RNN Based Approach for Bowel + Extravasation [Stage 2]

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2F81382ef287a512fd45d94e827c49e562%2FScreenshot%202023-10-22%20at%205.29.28%20AM.png?generation=1697935324304083&alt=media)

## **Data Preprocessing:**

The preprocessing involved taking patient/study scans, running a 3D segmentation model to output masks for each slice, and making study-level crops based on the boundaries of organs (liver, spleen, kidney, and bowel). 

We then create volumes from the patient, with each volume extracted as equidistant 96 slices for a study, reshaped to (32, 3, image_size, image_size) in a 2.5D manner for training CNN-based models. 

3 channels are formed by using the adjacent slices.

All our models take input in shape (2, 32, 3, height, width) and output in shape (2, 32, n_classes). The targets are also kept in shape (2, 32, n_classes).

The targets are derived by normalizing segmentation model masks in 0-1 based on the number of positive pixels, then multiplying the targets by the patient-level target for each middle slice of the sequence.

## **Stage 2: 2.5D Approach (2D CNN + RNN):**

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2Fe8df4581839b1fa7dcadf68fe2a715a1%2FScreenshot%202023-10-22%20at%205.31.23%20AM.png?generation=1697935695484067&alt=media)

In Stage 2, we trained our models using the volumes either based on our windowing or theo's preprocessing approach and the masks/crops generated from the 3D segmentation approach. Each model is trained for multiple tasks (segmentation + classification). For all 32 sequences, we predicted slice-level masks and sigmoid predictions. Simple maximum aggregation is applied to the sigmoid predictions to generate study-level predictions for submission.

### **Common Training Settings:**
- **Learning Rate:** (1e-4 to 4e-4) range
- **Optimizer:** AdamW
- **Scheduler:** Cosine Annealing with Warmup 
- **Loss:** BCE Loss for Classification, Dice Loss for Segmentation

### **Auxiliary Segmentation Loss:** 
We used auxiliary losses based on segmentation to stabilize training and improve scores. The encoder was shared between the classification and segmentation decoders, with two types of segmentation heads: a Unet-based decoder for generating masks and a 2D-CNN-based head. This approach improved our models by approximately +0.01 to +0.03 in performance.

Here’s an example code snippet for applying auxiliary loss:

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

The final ensemble for all organ

 models consisted of 7-8 diverse models. Study-level logits from these models were averaged to obtain the final prediction.

---

This version of the `README.md` focuses solely on the models and methods you have selected, with irrelevant information removed for clarity.