# Real-Time-X-Tweet-Sentiment-Analysis
Real-Time Streaming X(Tweet) Sentiment Analysis on Influencers

<div style="display: flex; flex-direction: column; align-items: center;">

# Real-Time Streaming X(Tweet) Sentiment Analysis on Influencers

## A Comprehensive Analysis of Influencer Sentiment Trends Using Apache Spark, Delta Lake, Databricks, and Hugging Face Transformer Models

---

## 📑 **Table of Contents**
1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Project Architecture](#3-project-architecture)
4. [Data Ingestion and Medallion Pipeline](#4-data-ingestion-and-medallion-pipeline)
5. [Methodology](#5-methodology)
   - [Bronze Data](#51-bronze-data)
   - [Silver Data](#52-silver-data)
   - [Gold Data](#53-gold-data)
6. [Model Inference and Evaluation](#6-model-inference-and-evaluation)
7. [Results](#7-results)
8. [Performance Optimization](#8-performance-optimization)
9. [Conclusions and Future Work](#9-conclusions-and-future-work)
10. [Acknowledgements and References](#10-acknowledgements-and-references)

---

## 1. Introduction

Influencers play a pivotal role in shaping public perception. By analyzing the sentiment of tweets directed at these figures, we can uncover trends and insights into public opinion. This project utilizes a real-time sentiment analysis pipeline, leveraging distributed data processing with Apache Spark, Delta Lake, and Databricks, alongside a Hugging Face transformer for sentiment classification.

---

## 2. Problem Statement

This project aims to:
- Build a scalable, real-time tweet sentiment analysis pipeline.
- Capture influencer sentiment trends and categorize them as positive, negative, or neutral.
- Identify patterns in influencer mentions and visualize sentiment shifts over time.

---

## 3. Project Architecture

The analysis follows a medallion data architecture:
- **Bronze Layer**: Raw tweet ingestion.
- **Silver Layer**: Preprocessed data with cleaned text and structured fields.
- **Gold Layer**: Sentiment inference results with enriched metadata.

Key technologies used:
- **Apache Spark** for distributed processing.
- **Delta Lake** for ACID-compliant storage.
- **Hugging Face Transformer** for sentiment classification.
- **MLflow** for tracking model performance.
- **Databricks** for streamlined development and execution.

---

## 4. Data Ingestion and Medallion Pipeline

### 4.1 Pipeline Overview
This end-to-end pipeline processes over 200,000 tweets, transforming raw data into structured, sentiment-enriched datasets.

### 4.2 Data Source
- Tweets were ingested in JSON format.
- Fields included:
  - `date`: Timestamp of the tweet.
  - `user`: Author of the tweet.
  - `text`: Tweet content.
  - `sentiment`: Pre-labeled sentiment (used for benchmarking).
  - `source_file`: JSON source file path.
  - `processing_time`: Time when the tweet was ingested.

---

## 5. Methodology

### 5.1 Bronze Data
The **Bronze Layer** stores raw ingested data without transformations.

#### Key Steps:
- **Schema Enforcement**: JSON schema defined to structure the data.
- **Streaming Input**: Databricks Auto Loader streams new JSON files into Delta Lake.
- **Data Columns**:
  - `date`, `user`, `text`, `sentiment`, `source_file`, `processing_time`.

---

### 5.2 Silver Data
The **Silver Layer** applies preprocessing and transformations.

#### Key Steps:
- **Timestamp Conversion**: String dates converted to timestamp format.
- **Mention Extraction**: Extracted `@username` mentions.
- **Text Cleaning**: Removed mentions and URLs from the tweet text.
- **Columns**:
  - `timestamp`, `mention`, `cleaned_text`, `sentiment`.

---

### 5.3 Gold Data
The **Gold Layer** enriches data with real-time sentiment predictions.

#### Key Steps:
- **Sentiment Model Inference**: Hugging Face sentiment transformer applied to `cleaned_text`.
- **Prediction Fields**:
  - `predicted_score`: Confidence score of sentiment prediction.
  - `predicted_sentiment`: Predicted label (`positive`, `negative`, `neutral`).
  - `sentiment_id` and `predicted_sentiment_id` for numerical representation.
  
---

## 6. Model Inference and Evaluation

### 6.1 Hugging Face Sentiment Model
- Pre-trained transformer loaded using **MLflow** for real-time inference.
- Model achieved **75% precision** on tweet sentiment classification.

### 6.2 Evaluation Metrics
- Precision: 0.75
- Recall: 0.66
- F1-score: 0.67

### 6.3 Confusion Matrix
| **True Sentiment** | **Predicted Positive** | **Predicted Negative** |
|-------------------|------------------------|------------------------|
| Positive           | 420                    | 120                    |
| Negative           | 200                    | 310                    |

Visualization:
![Confusion Matrix](confusion_matrix.png)

---

## 7. Results

### 7.1 Sentiment Analysis of Mentions
- **Total Mentions**: 64,742 unique mentions.
- **Sentiment Breakdown**:
  - Positive: 30%
  - Neutral: 35%
  - Negative: 35%

### 7.2 Top Influencers by Sentiment:
- **Top Positive Mentions**: @mileycyrus (417 positive mentions)
- **Top Negative Mentions**: @tommcfly (76 negative mentions)

### 7.3 Visualization
Interactive bar charts showing the top 20 influencers by sentiment are available for exploration.

---

## 8. Performance Optimization

The Spark pipeline was optimized to handle 200,000+ tweets efficiently:
- **Shuffle Partitions**: Set to `8` for parallelism.
- **Delta Table Optimizations**: Small files compacted using Delta Lake's `optimize()` and `compaction` operations.
- **Adaptive Query Execution**: Enabled for dynamic resource allocation.

---

## 9. Conclusions and Future Work

### Key Takeaways:
- The system effectively handles large-scale streaming data and provides real-time sentiment insights.
- Influencer sentiment can be tracked to identify public opinion trends.

### Next Steps:
- Extend the analysis to cover more social media platforms.
- Implement sentiment trend prediction using time series models.

---

## 10. References
1. [Apache Spark Documentation](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
2. [Databricks Autoloader Documentation](https://docs.databricks.com/spark/latest/structured-streaming/auto-loader.html)
3. [Hugging Face Model Hub](https://huggingface.co/models)

</div>
