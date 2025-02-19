# 🏡 Advanced Real Estate Valuation with Ensemble Regression Models  

## 📌 Overview  
This project aims to predict **real estate prices** using **ensemble regression models**, including **Random Forest Regressor** and **Gradient Boosting Regressor**.  
The models are trained using a dataset containing various features like **location, number of bedrooms, area,** and other relevant parameters.  

We have implemented this project using **Google Colab** for model training and **Django** for deploying the prediction system as a web application.  

---

## 🚀 Tech Stack Used  
- **🐍 Python** – for model training and evaluation  
- **📊 Google Colab** – for dataset analysis and model development  
- **🌐 Django** – for web framework and deployment  
- **🤖 Scikit-Learn** – for machine learning models  
- **📈 Pandas, NumPy** – for data manipulation  
- **📉 Matplotlib, Seaborn** – for visualization  
- **💾 Joblib** – for model saving/loading  

---

## 📂 Dataset  
The dataset contains various features such as:  
✔️ **House age**  
✔️ **Distance to the nearest MRT station**  
✔️ **Number of convenience stores nearby**  
✔️ **Latitude & Longitude**  
✔️ **House price per unit area (Target Variable)**  

🛠 **Preprocessing Steps:**  
- Handling missing values  
- Encoding categorical variables  
- Normalizing numerical features  

---

## 🔥 Model Training Process  

### 📌 **Data Preprocessing:**  
✅ Handling missing values  
✅ Encoding categorical variables  
✅ Feature scaling  

### 🏆 **Model Selection & Training:**  
✅ **Random Forest Regressor**  
✅ **Gradient Boosting Regressor**  
✅ Hyperparameter tuning using **GridSearchCV**  

### 📊 **Model Evaluation:**  
✅ Mean Squared Error (**MSE**)  
✅ Mean Absolute Error (**MAE**)  
✅ R² Score  

### 📈 **Visualization Techniques:**  
✅ **Pair plots**  
✅ **Correlation heatmaps**  
✅ **Residual plots**  
✅ **Model comparison graphs**  

---

## 🌍 Model Deployment using Django  
1️⃣ **User enters property details via a web form**  
2️⃣ **Pre-trained models predict the property value**  
3️⃣ **Results are displayed on the webpage**  

---

## 🛠 Installation and Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Sahana-S-Badiger/Advanced-Real-Estate-Valuation-with-Ensemble-Regression-Models.git
cd Advanced-Real-Estate-Valuation-with-Ensemble-Regression-Models
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Django Server
```bash
python manage.py runserver
```

🔗 Open http://127.0.0.1:8000/ in your browser to access the web application.

---

💡 Usage
1️⃣ Enter house details on the prediction page.
2️⃣ Click "Predict" to get the estimated price.
3️⃣ The results page displays the predicted value using trained ML models.

---

🔮 Future Enhancements
☁️ Deploy model on cloud platforms (AWS, Heroku, or GCP)
🧠 Use deep learning models for better accuracy
🎨 Enhance UI/UX for a better user experience

---
