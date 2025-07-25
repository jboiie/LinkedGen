
# LinkedGen: AI-powered LinkedIn Post Generator
### LinkedGen is a full-stack, AI-powered application designed to generate human-like, professional LinkedIn posts from user input. Built entirely with open-source components, it leverages a fine-tuned language model (DistilGPT2) to create content based on a user's specific scenario and desired tone. This project serves as a practical exercise in building and deploying a complete AI product, developing a wide range of skills across the machine learning lifecycle.

## Project Structure
- `app/` – Streamlit UI and app logic
- `model/` – Model training, inference, and utilities (all scripts here)
- `data/` – Data files and scripts (add your dataset here)
- `tests/` – Test scripts
- `docker/` – Dockerfile and deployment configs

## Dependencies
All dependencies for EDA, model training, and hyperparameter tuning are in `requirements.txt`:

```
streamlit
pandas
torch
transformers
datasets
optuna
scikit-learn
matplotlib
seaborn
notebook
```