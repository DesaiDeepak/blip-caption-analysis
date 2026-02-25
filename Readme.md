# Automated Caption Generation using BLIP Model

This project fine-tunes the [BLIP (Bootstrapping Language-Image Pre-training)](https://huggingface.co/Salesforce/blip-image-captioning-large) model on the Flickr8k dataset to automatically generate captions for images. A Streamlit web app is included for local deployment.

---

## Project Structure

```
TEAM5_AUTOMATED_CAPTION_GENERATION_USING_BLIP_MODEL/
│
├── Images/                        # Dataset folder (downloaded via script)
│   ├── Images/                    # Raw image files
│   ├── captions.txt               # Original captions file
│   └── cleaned_captions.txt       # Generated after data cleaning
│
├── cleaning/                      # Data cleaning scripts
│   └── data_cleaning.py
│
├── training/                      # Model training scripts
│   └── model_training.py
│
├── evaluation/                    # Model evaluation scripts
│   └── model_evaluation.py
│
├── saved_models/                  # Fine-tuned model saved here after training
│   └── fine_tuned_blip/
│
├── app.py                         # Streamlit web app
├── main.py                        # Main pipeline (clean → train → evaluate)
├── create_project_folders.py      # Sets up folders and downloads dataset
├── requirements.txt               # Python dependencies
└── .env                           # HuggingFace token (not committed)
```

---

## Setup Instructions

### Step 1 — Download the Dataset

Download the Flickr8k dataset from Kaggle:
[https://www.kaggle.com/datasets/ming666/flicker8k-dataset](https://www.kaggle.com/datasets/ming666/flicker8k-dataset)

Or run the setup script which handles it automatically:
```bash
python create_project_folders.py
```

---

### Step 2 — Create a Virtual Environment

```bash
conda create -n blip_env python=3.10
conda activate blip_env
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Set Up Environment Variables

Create a `.env` file in the root folder:
```
HF_TOKEN=your_huggingface_token_here
```

Get your token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

### Step 5 — Train the Model

```bash
python main.py
```

This will:
- Clean the captions file
- Fine-tune the BLIP model on Flickr8k
- Save the model to `saved_models/fine_tuned_blip/`
- Run evaluation (BLEU score)

---

### Step 6 — Run the Web App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to upload an image and generate captions.

---

## Requirements

- Python 3.10
- PyTorch 2.0.1
- Transformers 4.34.0
- Streamlit 1.21.0
- NLTK, Pillow, NumPy

See [requirements.txt](requirements.txt) for the full list.

---

## Team

Team 5 — Automated Caption Generation using BLIP Model
