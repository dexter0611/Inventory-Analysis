# 📊 Inventory Analysis with Blockchain Security

This project combines **AI/ML** and **Blockchain concepts** to analyze inventory data while ensuring **data accuracy** and **security**.  
It provides interactive data visualizations, predictive insights, and blockchain-based integrity verification.

---

## 🚀 Features
- **Column Selection**: Detect or manually specify the required columns (`ProductName`, `Cost`, `QuantitySold`).
- **Data Cleaning & Hashing**: Clean inventory data and generate a secure **SHA-256 hash**.
- **Blockchain Integration (Simulated)**: Store and verify data hashes to prevent tampering.
- **Aggregate Analysis**:
  - Calculate `TotalCost = Cost × QuantitySold`.
  - Group data by product for total cost analysis.
- **Visualizations**:
  - Top 5 and Bottom 5 products by total cost.
  - Interactive bar charts with **Plotly**.
- **Machine Learning**:
  - Train a **Random Forest Regressor** on `Cost` and `QuantitySold` to estimate `TotalCost`.
- **Dynamic Filtering**:
  - Slider to filter products by total cost in real-time.
- **Streamlit Interface**:
  - Easy-to-use web app for uploading CSV files and exploring results.

---

## 🛠️ Technologies Used
- [Streamlit](https://streamlit.io/) – Web app framework  
- [Pandas](https://pandas.pydata.org/) – Data handling  
- [Plotly](https://plotly.com/python/) – Interactive visualizations  
- [Scikit-learn](https://scikit-learn.org/) – Machine learning (Random Forest Regressor)  
- [Hashlib](https://docs.python.org/3/library/hashlib.html) – Secure data hashing  
- Blockchain (simulated) – Ensures data integrity  

---

## 📂 Project Structure
