University Chatbot Project
=========================

This document explains the repository contents, illustrates how each file works, and describes the full data flow from raw intent definitions to the deployed web chatbot.

Project Summary
---------------
- A university query chatbot built using intent classification.
- Uses `dataset/intents.json` as the training dataset.
- Preprocesses text, trains machine learning models, selects the best model, and deploys a Flask-based web UI.
- Supports direct phrase matching for a few common queries and fallback behavior for unknown questions.

Repository Structure
--------------------
- `app.py` - Flask web application that serves the UI and exposes the `/predict` endpoint.
- `chatbot.py` - Main inference engine that loads preprocess artifacts, selects the best trained model, performs text preprocessing, predicts intent, and returns a response.
- `preprocess_data.py` - Data preprocessing pipeline: loads intents, cleans text, tokenizes/lemmatizes, builds training/testing splits, and saves text/vector artifacts.
- `train_models.py` - Trains classification models, evaluates them, saves model files, and writes metrics.
- `plot_results.py` - Generates comparison charts and confusion matrices from saved metrics and results.
- `test_api.py` - Simple API test script using Flask test client.
- `dataset/intents.json` - Intent definitions, training phrases, and response templates.
- `results/` - Saved model artifacts, metrics, confusion matrices, and best model information.
- `templates/index.html` - Client-side chat interface and browser-side request logic.
- `static/style.css` - Chat UI styling.

How the System Works
--------------------

1. `dataset/intents.json`
   - Contains a top-level `intents` list.
   - Each intent object includes:
     - `intent` (intent label)
     - `text` list (example user messages)
     - `responses` list (possible bot replies)
   - Other fields like `extension`, `context`, and `entities` exist but are not used by the current pipeline.

2. `preprocess_data.py`
   - Loads the JSON dataset.
   - Normalizes and cleans each text example:
     - removes HTML tags
     - converts to lowercase
     - removes punctuation and non-alphanumeric characters
     - collapses whitespace
   - Tokenizes the cleaned text using NLTK if available, otherwise simple splitting.
   - Filters stop words and words shorter than 2 characters.
   - Lemmatizes tokens with NLTK's `WordNetLemmatizer`; if that fails, falls back to Porter stemming.
   - Builds a dataset of processed text and labels.
   - Writes:
     - `training_data.csv`
     - `testing_data.csv`
     - `vectorizer.pkl` (TF-IDF vectorizer)
     - `label_encoder.pkl` (label encoder)

3. `train_models.py`
   - Loads `training_data.csv` and `testing_data.csv`.
   - Loads `vectorizer.pkl` and `label_encoder.pkl`.
   - Transforms text into TF-IDF features.
   - Trains the following models:
     - `MultinomialNB` (Naive Bayes)
     - `RandomForestClassifier`
     - `NeuralNet` (TensorFlow Keras, only if TensorFlow is installed)
   - For each model:
     - performs training
     - computes training/test accuracy
     - computes precision, recall, and F1 score
     - generates a confusion matrix
   - Saves:
     - `results/model_nb.pkl`
     - `results/model_rf.pkl`
     - `results/model_nn.h5` (if TensorFlow available)
     - `results/best_model.pkl` or `results/best_model_info.json`
     - `results/metrics.json`
     - `results/confusion_<model>.npy`

4. `chatbot.py`
   - Loads required artifacts at startup:
     - `vectorizer.pkl`
     - `label_encoder.pkl`
     - best saved model from `results/best_model.pkl` or `results/model_nn.h5`
   - Loads `dataset/intents.json` to map intent tags to response sets.
   - Stores a built-in direct phrase map (`direct_intents`) for exact normalized queries like greetings, goodbyes, and a handful of exam/result questions.
   - Prediction flow:
     - Normalize raw text.
     - If the normalized phrase matches `direct_intents`, select that known tag and a random response.
     - Otherwise preprocess text using the same cleaning/tokenization/lemmatization logic as training.
     - Transform the processed text with the vectorizer.
     - Use the loaded model to predict an intent.
     - If the model is a scikit-learn model, derive confidence from `predict_proba`.
     - If the model is a Keras model, use predicted softmax probabilities.
     - If confidence is below `0.30` or no responses exist for the predicted tag, return a fallback reply.

5. `app.py`
   - Creates a Flask app.
   - Serves the chat page at `/` using `templates/index.html`.
   - Accepts POST requests at `/predict` with JSON `{message: text}`.
   - Uses `ChatbotEngine.predict()` and returns JSON with `intent` and `response`.

6. `templates/index.html`
   - Simple client UI with a chat history area and message entry field.
   - Sends messages using `fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:text}) })`.
   - Displays user and bot messages as chat bubbles.
   - Includes a debug panel that logs browser-side activity to localStorage for easier troubleshooting.
   - Adds global `error` and `unhandledrejection` handlers so client failures appear in the chat UI.

7. `static/style.css`
   - Styles the chat window and bubbles.
   - Defines the UI layout, colors, spacing, and responsive behavior.

8. `test_api.py`
   - Uses Flask's test client to call `/predict` with sample messages.
   - Prints the status code and returned JSON for quick sanity checks.

How to Run the Project
----------------------
1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Preprocess the dataset:

```bash
python preprocess_data.py
```

4. Train models:

```bash
python train_models.py
```

5. Start the web app:

```bash
python app.py
```

6. Open a browser at `http://127.0.0.1:5000`.

7. Optionally run the API test script:

```bash
python test_api.py
```

Important Notes
---------------
- The model pipeline depends on `nltk` for tokenization and lemmatization.
- If TensorFlow is unavailable, the neural network model is skipped and only scikit-learn models are trained.
- The direct phrase map in `chatbot.py` improves responses for simple exact-match questions by bypassing the classifier.
- The project stores trained model artifacts in `results/` and runtime artifacts in the repo root (`vectorizer.pkl`, `label_encoder.pkl`).

What I Have Done
-----------------
- Inspected the repository structure and code files.
- Reviewed the data preprocessing, model training, inference, and web UI behavior.
- Created `README.d` with a complete explanation of each file, data flow, and running instructions.
- Documented both the training pipeline and the runtime prediction flow, including fallback behavior.

This file is intended to serve as an expanded project README and design overview for the entire repository.