# Cybersecurity Attacks Data Science Pipeline

A complete end‑to‑end data science pipeline for exploring, analyzing, and modeling a 40,000‑record cybersecurity attacks dataset from Kaggle. This project demonstrates how to go from raw logs to actionable insights and predictive models.

##  Repository Structure

- **data/** – raw and processed CSVs  
- **scripts/** – Python scripts for each pipeline stage  
- **results/** – saved charts, graphs & model outputs  
- **notebooks/** – exploratory analyses & prototyping  

##  Pipeline Stages

1. **Data Ingestion & Cleaning**  
   - Loaded 25‑field CSV, handled missing values, dropped irrelevant columns.  
2. **Metadata & Statistics**  
   - Profiled data types, missing rates, basic distributions, correlations.  
3. **Anomaly Detection**  
   - Used Isolation Forest to flag outliers in numeric features.  
4. **Clustering & Segmentation**  
   - PCA‑reduced data → KMeans (k=3) → interpreted clusters by traffic patterns.  
5. **Segment Analysis**  
   - Computed per‑segment statistics (packet length, port usage) and temporal trends.  
6. **NLP on Alerts**  
   - Tokenized “Alerts/Warnings” text, generated word cloud, observed neutral sentiment.  
7. **Graph Visualization**  
   - Sampled host‑to‑host edges → NetworkX graph → visualized information flow.  
8. **Modeling**  
   - Regression (Random Forest) for anomaly scores, classification (RF) for severity levels  
   - Evaluated with cross‑validation, feature importances, and confusion matrix.  
9. **Reporting & Recommendations**  
   - Summarized findings, suggested next steps (temporal drift, federated learning, GNNs, etc.)
