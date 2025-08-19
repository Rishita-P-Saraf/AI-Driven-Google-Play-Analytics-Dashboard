# 📊 Google Play Store Data Analysis (Interactive Dashboard with Plotly)

## Internship Project — NullClass Data Analyst Intern

## 📌 Project Overview

This project analyzes Google Play Store data to uncover insights about app performance, user sentiment, and monetization trends.
It combines data cleaning, transformation, visualization, and NLP-based sentiment analysis, presented in an interactive Plotly dashboard deployed online.

---

## 🛠️ Tech Stack

- Python Libraries: Pandas, NumPy, Plotly, Scikit-learn, NLTK (VADER)
- Visualization: Interactive dashboards with Plotly
- NLP: Sentiment analysis using VADER
- Deployment: Netlify
- Others: WordCloud, Time-based Filtering

---

## 📊 Features & Tasks
### 1. Word Cloud for Health & Fitness Apps
- Generate a word cloud for frequent keywords in 5-star reviews
- Exclude common stopwords and app names
- Filter reviews only for apps in the "Health & Fitness" category

### 2. Grouped Bar Chart for App Categories
- Compare the average rating and total review count for the top 10 app categories by installs
- Apply filters:
  - Exclude categories with an average rating below 4.0
  - App size must be at least 10MB
  - Last update should be in January
  - Display the graph **only between 3 PM IST to 5 PM IST**

### 3. Bubble Chart for Games Category
- Analyze the relationship between app size and average rating
- Bubble size represents the number of installs
- Apply filters:
  - Rating should be above 3.5
  - Only include "Games" category
  - Installs should be more than 50,000
  - Time-based filter: Display only between 5 PM – 7 PM IST

---

## 📈 Dashboard Highlights
- **Dynamic Visuals**: Interactive and time-sensitive charts
- **User Insights**: Sentiment analysis to understand user feedback
- **Category Trends**: Identifying top app genres and performance metrics
- **Revenue Analysis**: Understanding monetization trends

---

## 📂 Project Structure
```sql
Google-PlayStore-Data-Analysis/
│
├── data/
│   ├── Play Store Data.csv
│   └── User Reviews.csv
│
├── notebooks/
│   └── Google PlayStore Data Analysis using Plotly.ipynb
│
├── scripts/
│   └── Google PlayStore Data Analysis using Plotly.py
│
├── LICENSE
├── README.md

```

---

## 📌 Key Learnings

- Data wrangling with Pandas & NumPy
- Building interactive dashboards with Plotly
- Sentiment Analysis (NLP) using NLTK VADER
- Handling time-based filtering & conditions in visualizations
- Deploying dashboards with Netlify

---

## 📸 Screenshots


<img width="998" height="519" alt="Screenshot 2025-08-19 125148" src="https://github.com/user-attachments/assets/55c9b814-9629-498e-b865-f8e3d133a0cf" />

<img width="1895" height="882" alt="Screenshot 2025-08-19 122753" src="https://github.com/user-attachments/assets/538d0928-0467-4b9e-8ecb-a5ac9dd168f5" />

<img width="1900" height="1079" alt="image" src="https://github.com/user-attachments/assets/1409f346-be54-4ed3-beab-c777456a20de" />

---

## Conclusion
This internship project provided hands-on experience in data analysis, visualization, and NLP.
The interactive dashboard delivers actionable insights, making it a valuable tool for app developers and business analysts.
---

