# MDAMA
MDAMA: Multi-Source Domain Adaptation with Mixture Alignment 

### Setting Up the Project

Follow these instructions to set up the project environment and data.

#### 1. Create a New Conda Environment

To isolate your project dependencies, create a new Conda environment. Open your terminal and run the following command:

```bash
conda create --name mdama python=3.10.10
```
#### 2. Activate the Conda Environment

```bash
conda activate mdama

```
#### 3. Install Required Packages

```bash
pip install -r requirements.txt
```
#### 4. Download the Office Home Dataset
Download the Office Home Dataset from the official website (https://www.hemanthdv.org/officeHomeDataset.html). 
Once downloaded, organize the dataset by placing it in the /data folder. The directory structure should resemble this:

```bash
data/
│
├── Art/
│   ├── AlarmClock/
│   ├── ...
│
├── Clipart/
│   ├── ...
│
├── Product/
│   ├── ...
│
├── RealWorld/
│   ├── ...
│
└── imagelist/
    ├── ...
```
#### 5. 
Ensure you do not make any modifications to the "imagelist" folder. This folder contains paths to images across all domains.

### Dependency

```bash
higher==0.2.1
matplotlib==3.7.1
mmcv==2.0.1
numpy==1.24.3
Pillow==9.5.0
prettytable==3.7.0
scikit_learn==1.2.2
scipy==1.10.1
timm==0.9.2
torch==2.0.1
torchvision==0.15.2
tqdm==4.65.0
```

### Train

- [x] MDAMA on `Office-Home` dataset:
     The following command is provided for utilizing the source domains 'Cl, Pr, Rw,' along with the target domain 'Ar.' For instructions pertaining to other domains, please refer to the respective mdama.sh file.

  ```bash
   CUDA_VISIBLE_DEVICES=0 python mdama_OH.py data/office-home -d OfficeHome -s Cl Pr Rw -t Ar -a resnet50 epochs 30 --bottleneck-dim 2048 --seed 0 --log logs/mdama_OH/OfficeHome_:2Ar
  ```
### Output
 - [x] Upon the completion of training for 40 epochs, the metric 'test_acc1'is also shown as an indicator of test accuracy.
