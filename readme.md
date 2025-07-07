# 🏆 AVI Challenge 2025 - Track1: Personality Assessment

## 🎯 Project Overview
This project implements a deep learning model based on multimodal feature fusion that effectively integrates audio, video, and text information to predict the following personality traits:
- **Honesty-Humility** 
- **Extraversion** 
- **Agreeableness** 
- **Conscientiousness** 

**We achieved 1st place in AVI Challenge 2025 Track1: Personality Assessment!** This repository contains our code structure.
| **Team Name**       | **MSE (↓)** |
|---------------------|--------------------|
| **HFUT-VisionXL**    | **0.12542 (1)**    |
| Jezoid               | 0.13724 (2)        |
| CAS-MAIS             | 0.14351 (3)        |
| The innovators       | 0.14492 (4)        |
| ABC-Lab              | 0.16770 (5)        |
| Winner-Team          | 0.18909 (6)        |
| HSEmotion            | 0.19731 (7)        |
| abhisheksingh        | 0.19779 (8)        |
| DERS                 | 0.20612 (9)        |
| DERS                 | 0.20674 (10)       |
| USTC-IAT-United      | 0.22914 (11)       |
| HandX                | 0.23824 (12)       |
| wjno1                | 0.24358 (13)       |
| gkdx2                | 1.89703 (14)       |


## 🏗️ Project Structure

```
AVI-track1/
├── train_task1.py          # Training script
├── test_task1.py           # Testing script
├── readme                  # Project documentation
├── model/                  # Model implementation
│   ├── builder.py          
│   ├── ATconnector.py      
│   ├── VTconnector.py      
│   ├── TextFeatureEnhancer.py  
│   └── Token_Refinement_Module.py  
├── dataset/                # Dataset processing
│   └── baseline_dataset.py 
├── Features/               # downloaded features
│   ├── audio/               # Audio features
│   ├── video/               # Video features
│   └── text/                # Text features
├── scripts/                # Execution scripts
│   ├── train.sh            # Training script
│   └── test.sh             # Testing script
├── args_log/
├── data/
├── results/
├── save_ckpt/              # Model checkpoints
├── preprocess/             # Data preprocessing
└── requirements.txt        # Project dependencies
```

## ⚙️ Environment Requirements
```bash
git clone url_to_your_repository
cd AVI-track1
```

### Step 1: Create a new conda environment with Python 3.10 (or your preferred version)
conda create -n avi2025 python=3.10 -y

### Step 2: Activate the environment
conda activate avi2025

### Step 3: Install pip (if not already installed)
conda install pip

### Step 4: Install dependencies from requirements.txt
python -m pip install -r requirements.txt

## 🚀 Quick Start


### 1. 📋 Data Preparation

Downloaded features are required for training and testing. The dataset includes audio, video, and text features extracted from the AVI Challenge 2025 dataset.
Baidu Cloud link for downloading the features: [Baidu Cloud](https://pan.baidu.com/s/1J2b0g3k4Z5a9d8e9f8g88g) (Password: `avi2025`)

Ensure data file paths are correctly configured:
```bash
# Training data
TRAIN_CSV="path/to/train_data.csv"
VAL_CSV="path/to/val_data.csv"
TEST_CSV="path/to/test_data.csv"

# Feature directories
AUDIO_DIR="path/to/audio/features"
VIDEO_DIR="path/to/video/features"
TEXT_DIR="path/to/text/features"
```

### 2. 🏋️‍♂️ Model Training

Use the provided training script:
```bash
cd AVI-track1
bash ./scripts/train.sh
```

### 3. 🧪 Model Testing
Note: To achieve better generalization performance, we recommend that only text features be used during testing.
```bash
bash ./scripts/test.sh
```
Ensure data file paths are correctly configured:
```bash
# Training data
ARGS_JSON="path/to/args.json"
SUBMISSION_CSV="path/to/submission.csv"
TEST_CSV="path/to/test_data.csv"

# trait to predict 
NOTE: trait should be same as label_col in args.json
TRAIT="Honesty-Humility"  # Honesty-Humility, Extraversion, Agreeableness, Conscientiousness
```

### 📏 Loss Function
-  Mean Squared Error (MSE) loss


## 📋 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

If you have any questions or suggestions, please contact the project maintainers (HFUT-VisionXL).

---

⚠️ **Note**: This project is for academic research purposes only. Please comply with relevant data usage agreements and competition rules.

## 🙏 Acknowledgments

- 🏆 Thanks to the AVI Challenge 2025 organizers
- 🤗 Thanks to the developers of [MERtools](https://github.com/zeroQiaoba/MERTools) for their excellent open-source tools that supported our data preprocessing.