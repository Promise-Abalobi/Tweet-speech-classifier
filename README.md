# 🗣 Speech Classifier

A machine learning web application for classifying Twitter tweets as **Hate Speech**, **Offensive Language**, or **None**. Built with Python, scikit-learn, and Flask.

## 🚀 Features

- **Text Classification**: Detects hate speech and offensive language in user-inputted text.
- **Web Interface**: User-friendly web app for real-time predictions.
- **Pre-trained Model**: Uses a Support Vector Classifier (SVC) trained on a labeled Twitter dataset.
- **Interactive Visualization**: Includes model evaluation metrics and confusion matrix.

## 🏗️ Project Structure

```
App.py
cv.pkl
model.pkl
requirements.txt
speechclf.ipynb
templates/
    index.html
```

- `App.py`: Flask web server for prediction API and web UI.
- `cv.pkl`: Saved CountVectorizer for text preprocessing.
- `model.pkl`: Trained SVC model for classification.
- `requirements.txt`: Python dependencies.
- `speechclf.ipynb`: Jupyter notebook for data cleaning, training, and evaluation.
- `templates/index.html`: Web UI template.

## 🖥️ Demo

1. Run the Flask app:
    ```sh
    python App.py
    ```
2. Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000)
3. Enter a sentence and click **Predict** to see the classification result.

## 📦 Installation

1. **Clone the repository**  
    ```sh
    git clone <your-repo-url>
    cd speech-clf
    ```

2. **Install dependencies**  
    ```sh
    pip install -r requirements.txt
    ```

3. **(Optional) Retrain the model**  
    Open `speechclf.ipynb` and run all cells to retrain and save `model.pkl` and `cv.pkl`.

## 📝 Usage

- **Web App**: Use the web interface to classify any text.
- **Jupyter Notebook**: Explore data preprocessing, model training, and evaluation.

## 🧠 Model Details

- **Algorithm**: Support Vector Classifier (SVC) with linear kernel.
- **Vectorization**: Bag-of-words using scikit-learn's `CountVectorizer`.
- **Classes**:
    - `Hate Speech`
    - `Offensive Language`
    - `None`

## 📊 Evaluation

- **Accuracy**: ~90%
- **Metrics**: Precision, Recall, F1-score, Confusion Matrix (see `speechclf.ipynb` for details).

## 🛠️ Technologies

- Python 3
- Flask
- scikit-learn
- pandas, numpy
- Jupyter Notebook

## 📄 License

MIT License

---

*Created for educational purposes. Not intended for production use without further validation
