
# LinkedGen: AI-powered LinkedIn Post Generator

## Project Structure
- `app/` – Streamlit UI and app logic
- `model/` – Model training, inference, and utilities (all scripts here)
- `data/` – Data files and scripts (add your dataset here)
- `tests/` – Test scripts
- `docker/` – Dockerfile and deployment configs

## Setup
1. Create a Python virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

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

## Next Steps
- Add your LinkedIn post dataset to `data/`
- Use scripts in `model/` for training and tuning distilGPT2
- Build the Streamlit UI in `app/`
- Write tests in `tests/`
- Containerize with Docker

## Deployment
See `docker/` for Dockerfile and deployment instructions.
