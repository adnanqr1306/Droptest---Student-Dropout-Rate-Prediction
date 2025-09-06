# 🎓 Student Dropout Predictor

A machine learning web application built with **Streamlit** that predicts student dropout risk based on academic, demographic, and family-related factors. The project compares multiple classification models and provides visual insights into their performance.



## 🚀 Features

- 📊 **Model Performance Comparison** – Accuracy, Sensitivity, Specificity, F1-Score  
- 🔍 **Best Model Selection** – Automatically selects the top-performing model  
- 🎨 **Interactive Dashboard** – Clean UI with confusion matrix, ROC curve, and styled dataframes  
- 📝 **Student Input Form** – Collects student details and predicts dropout probability in real-time  
- 📈 **Probability Visualization** – Displays dropout vs staying probabilities with progress bars  



## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)  
- **Backend:** Python 3.12  
- **Libraries:**  
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `joblib`  
  - `matplotlib`, `seaborn`  



## 📂 Project Structure

```
.
├── data/
│   ├── processed/          # Train-test splits
│   └── test_set.csv        # Test dataset
├── all_models/                 # Trained ML models
│   └── *.pkl
├── src/
│   ├── app.py              # Streamlit application
│   ├── data_processing.py  # Data preprocessing scripts
│   └── ...
├── best_model/
│   └── best_model.pkl      # Best performing model
├── results/
│   └── model_metrics.csv     # All models datas
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```


## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/adnanqr1306/Droptest---Student-Dropout-Rate-Prediction.git
   cd Droptest---Student-Dropout-Rate-Prediction
   ```

2. **Create Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   venv\Scripts\activate      # For Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Streamlit App**
   ```bash
   streamlit run src/app.py
   ```



## 📊 Model Evaluation

During training, multiple machine learning models were tested, including:

- Logistic Regression  
- Random Forest  
- Gradient Boosting  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  

Metrics evaluated:  
- **Accuracy**  
- **Sensitivity (Recall)**  
- **Specificity**  
- **F1-Score**  
- **AUC-ROC Curve**  

The **best-performing model** is stored as `best_model/best_model.pkl` and is automatically loaded by the app.



## 🖼️ Screenshots

### Model Comparison
![Model Comparison](Droptest/screenshots/Droptest%20Model%20Comparison.png)

### Confusion Matrix
![Confusion Matrix](Droptest/screenshots/Droptest%20Confusion%20Matrix.png)

### ROC Curve
![ROC Curve](Droptest/screenshots/Droptest%20ROC%20Curve.png)

### Prediction Dashboard
![Prediction](Droptest/screenshots/Droptest%20Student%20Prediction.png)



## 🔮 Future Improvements

- Add more student-related features (attendance %, parental income, etc.)  
- Implement deep learning models for better accuracy  
- Deploy app on **Streamlit Cloud / Heroku / AWS**  

---

