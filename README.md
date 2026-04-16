 🍎 Food Nutrition AI (50K Dataset Version)

## 📌 Overview
This project is an advanced version of the Food Nutrition AI system that uses Natural Language Processing (NLP) and Machine Learning for food analysis.

This version uses a **large dataset of 50,000 food records**, which improves model accuracy but increases loading time.

## 🔁 Previous Version (Smaller Dataset)

This project is an extended version of our earlier implementation:

👉 GitHub Repository:  
https://github.com/stephygrace-2k5/Food-Nutrition-AI.git

### 📊 Details of Previous Version
- Dataset size: **2,395 records**
- Faster loading time
- Same system architecture and features

### 📂 Additional Resources (Available in Previous Version)
The previous repository also includes:
- 📊 PPT Presentation  
- 📄 Conference Paper  
- 📘 Word Document  

👉 These documents explain the complete system using the smaller dataset.

--

## ⚡ Current Version (50K Dataset)

This version expands the dataset to **50,000 records**, which improves:
- Model performance
- Prediction accuracy
- Data coverage

However:

⏳ **Initial loading may take 20–30+ minutes**  
due to large-scale data processing and NLP model building.

---

## 🧠 Note

Both versions use the same:
- Architecture
- AI models
- Workflow

👉 The only difference is **dataset size (2K vs 50K)**.

---

## ⚠️ Important Note (Loading Time)

⏳ This project may take **20–30+ minutes to load initially**.

### Why?
- The dataset contains **50,000 records**
- The system builds:
  - TF-IDF NLP index
  - Machine Learning models
- These processes are computationally heavy

👉 Please be patient during loading.

---

## 📊 Dataset Information

- Total records: **50,000**
- Data is split into 5 files:
  - FOOD-DATA-GROUP1.csv
  - FOOD-DATA-GROUP2.csv
  - FOOD-DATA-GROUP3.csv
  - FOOD-DATA-GROUP4.csv
  - FOOD-DATA-GROUP5.csv

---

## 🔁 Comparison with Previous Project

| Feature | Previous Version | This Version |
|--------|----------------|-------------|
| Dataset Size | 2,395 records | 50,000 records |
| Performance | Faster | Slower (initial load) |
| Accuracy | Moderate | Higher |
| Loading Time | Few seconds | 20–30+ minutes |

👉 This project is the **extended version** of the earlier Food Nutrition AI project with a significantly larger dataset.

---

## 📂 Additional Resources

The following are available in the previous version of this project:
- 📊 PPT Presentation
- 📄 Conference Paper
- 📘 Word Document

👉 These are based on the **smaller dataset (2,395 records)** but explain the same system.

---

## ▶️ How to Run

### Step 1: Install dependencies
```bash
pip install -r requirements.txt
````

### Step 2: Run the application

```bash
python app.py
```

### Step 3: Open in browser

```plaintext
http://127.0.0.1:5000
```

---

## ⚡ Optional (Faster Testing)

If you want faster loading for testing:

Edit `app.py` and add:

```python
df = df.sample(3000)
```

👉 This reduces dataset size and loads quickly.

---

## 🧠 Technologies Used

* Python
* Flask
* Scikit-learn
* Pandas, NumPy
* NLP (TF-IDF)

---

## 🎯 Key Features

* 🔍 NLP-based food search
* 📊 Calorie prediction
* 🧠 Classification models
* 📉 Clustering analysis
* 📈 Interactive visualizations

---

