# University Chatbot

Simple university query chatbot using NLP intent classification.

Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Preprocess data:

```bash
python preprocess_data.py
```

3. Train models:

```bash
python train_models.py
```

4. Plot results:

```bash
python plot_results.py
```

5. Run the web app:

```bash
python app.py
```

Files

- `preprocess_data.py` — preprocessing and data split
- `train_models.py` — trains Naive Bayes, RandomForest, and a small NN
- `plot_results.py` — generates comparison charts and confusion matrices
- `chatbot.py` — inference engine
- `app.py` — Flask web UI
- `templates/index.html`, `static/style.css` — UI
- `results/` — saved models and plots
