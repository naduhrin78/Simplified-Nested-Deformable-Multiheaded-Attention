# Simplified Nested Deformable Multi-head Attention for Facial Image Inpainting
![architecture](https://github.com/naduhrin78/Simplified-Nested-Deformable-Multiheaded-Attention/assets/19820757/163609ef-112a-42aa-a2c7-09cc0f7e1d84)

## Prerequisites
- Python 3.6+
- [PyTorch>1.0](https://pytorch.org/get-started/previous-versions/)
- cv2, numpy, PIL, torchvision
## Usage

Keep your dataset by placing images like:
dataset
    ├── CelebA-HQ
    │   ├── 1.png 
    │   ├── 2.png 
    │   │   └── ...
    ├── irregular_masks
    │   ├── 1.png 
    │   ├── 2.png 
    │   └── ...    

## Checkpoints:

Download the checkpoint: [![Checkpoints](https://img.shields.io/badge/Checkpoint-<COLOR>.svg)](https://drive.google.com/file/d/1mjO85DdatC_gg1ppNbqnNOlq8CLq_ih2/view?usp=sharing)

    The checkpoints are provided for:
        CelebA-HQ dataset
        Keep the checkpoints in root directory "./"


To test the network:
    
    python test.py
        


The results will be stored in:

    ./outputs
