
This is the procedure of the model for training and deploying the model inyto the local environment

# Download the dataset

https://www.kaggle.com/datasets/ming666/flicker8k-dataset


# Run create_project_folders.py file

that will create the needed achitecture for the project
Please place the files in the necessary stryure for the project running
 
 Imagecaptioning/
│
├── Images/                               # Data folder for raw and processed data
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│
├── saved_models/                             # Folder for language models
│   ├── fine_tuned_meta_llama/          
│   ├── pretrained_miniLM/              
│   └── model_config.json               # Configuration for fine-tuned models
│
│
├── cleaning/                            # cleaning scripts for specific tasks
│   ├── dataset.py                         # Data preprocessing script
│
├── data/                          # Jupyter notebooks for experiments
│   ├── data_exploration.ipynb          # Notebook for initial data exploration
│
├── training/                              #  tests for different modules
│   ├── model_training.py           # Unit tests for preprocessing
│
├── evaluation/                               # model evaluation and score
├── model_evaluation.py                       # mdoel evaluation python file
│
├── requirements.txt                    # Python dependencies
├── main.py                            # Script for setting up the project
├── README.md                           # Project overview and instructions
└── app.py                         # stramlit app   file
├── captions.txt                    # Captions file with image descriptions
└── cleaned-captions.py                         # genrated after data cleaneing for clearing the anamolies

 
 
Step1: Model Training

Running Steps:

# create a virtual environment


# command to create the virtual env
conda create -n venv python=3.10

note: venv --> my virtual env name
python=31.0 --->python version

replace the version and virtual envoironment name if needed  

# command to actiavte the virtual env
conda activate 


# Step2: Install all the required libraries

# command :   pip install -r requirements.txt

# step3 : Model Training and data cleaning

# command: python main.py

or run main.py it will call the folder structure of the file and train the model and save the model into the space


# step4 : call the save model or deploy into local environment


command: streamlit run app.py




