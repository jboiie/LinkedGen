# LinkedGen: AI-Powered LinkedIn Post Generator

LinkedGen is an AI application that generates professional, human-like LinkedIn posts from simple user input, leveraging a fine-tuned DistilGPT2 model. It includes a clean data pipeline, a working Streamlit UI, and a reproducible training workflow.

## Features
- Fine-tuned DistilGPT2 for LinkedIn-style post generation using Hugging Face Transformers.
- Streamlit UI: input a scenario, select a tone (e.g., humble, excited, grateful, sad, motivating, regretful), and generate a copy-ready post.
- Data pipeline: CSV loading, text cleaning, and reproducible train/validation split.
- Optional Dockerized deployment for consistent local and cloud runs.

## Project Motivation
- I built LinkedGen to gain hands-on, practical skills across the ML lifecycle—curating and cleaning a custom dataset, fine-tuning DistilGPT2 on Colab with saved artifacts, and integrating the model into a working Streamlit app for inference. The project gave me end-to-end practice from data processing and EDA to model training and UI integration, with clear next steps for Dockerization, simple evaluation/tracking, CI, and deployment to further solidify real-world engineering skills.

## Project Structure
```
├─ app/ # Streamlit app
│ └─ app.py
├─ data/ # Raw/processed data (train.csv, val.csv)
├─ models/ # Saved fine-tuned model (e.g., distilgpt2-finetuned)
├─ model/ # Training scripts (tuning/finetune)
│ ├─ finetune_distilgpt2.py
│ └─ tune_hyperparams.py
├─ src/ # Data processing utilities
│ └─ data_processing.py
├─ docker/ # Dockerfile and deployment assets (optional)
├─ requirements.txt # Python dependencies
└─ README.md
 ```
### Prerequisites
- Python 3.9+ and pip
- Recommended: virtual environment (venv/conda)
- (Optional) Docker

## Acknowledgments
- Hugging Face Transformers and Datasets
- Streamlit
- PyTorch

