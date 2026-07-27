# CampusAI

CampusAI is a Flask-based university assistant that answers common questions
about admissions, programs, fees, exams, facilities, and student services.

## Run the app

Python 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. The included trained model is ready to use; data
preprocessing and retraining are not required to run the web app.

You can change the port with the `PORT` environment variable. Debug mode is off
by default; set `FLASK_DEBUG=1` when developing.

## Verify the app

```powershell
python test_api.py
```

The check covers the prediction endpoint, empty-input validation, and the health
endpoint.

## Model development (optional)

Install the extra data and plotting packages:

```powershell
python -m pip install -r requirements-dev.txt
python preprocess_data.py
python train_models.py
python plot_results.py
```

`train_models.py` trains the scikit-learn models without TensorFlow. If
TensorFlow is installed separately, it also trains the optional neural network.

## Project layout

- `app.py` — Flask routes and request validation
- `chatbot.py` — model loading and inference
- `dataset/intents.json` — training phrases and responses
- `templates/index.html` — chat interface and interactions
- `static/style.css` — responsive visual design
- `results/` — trained models and evaluation output
