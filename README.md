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
10. [References](#10-references)

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

<img width="527" alt="Screenshot 2025-01-03 at 1 44 32 PM" src="https://github.com/user-attachments/assets/02b648d0-8f2e-43b8-bd39-afb39c217118" />

<img width="747" alt="Screenshot 2025-01-03 at 1 42 58 PM" src="https://github.com/user-attachments/assets/40e0db47-2d96-44b2-9e33-c0d70b1db424" />

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

<img width="895" alt="Screenshot 2025-01-03 at 1 44 50 PM" src="https://github.com/user-attachments/assets/93096250-0ea9-44d1-ad6f-667ba579e499" />

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

![image](https://github.com/user-attachments/assets/1aed2ce1-3ce0-4be0-8e0d-a8564ee99410)


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

  <img width="868" alt="Screenshot 2025-01-03 at 1 48 52 PM" src="https://github.com/user-attachments/assets/5e45361c-9b2c-4242-b72f-d1ed80cd721c" />

- **Top Negative Mentions**: @tommcfly (76 negative mentions)

  <img width="852" alt="Screenshot 2025-01-03 at 1 49 07 PM" src="https://github.com/user-attachments/assets/cfa86f1f-d192-4206-91c1-eef10876e1d2" />


### 7.3 Visualization
Interactive bar charts showing the top 20 influencers by sentiment are available for exploration.

<img width="806" alt="Screenshot 2025-01-03 at 1 47 50 PM" src="https://github.com/user-attachments/assets/29b23575-6b95-43e2-93f5-39617db1bacf" />



---
## **8. Performance Optimization**

The Spark pipeline was optimized to handle **200,000+ tweets** efficiently. Below is a breakdown of the performance dimensions evaluated using **Spark UI metrics** and additional tuning strategies applied.

### **8.1 Key Optimizations Applied**
- **Shuffle Partitions**: Set to `8` for parallelism in line with the available number of cores.
- **Delta Table Optimizations**: Small files compacted using Delta Lake's `optimize()` and `compaction` operations to reduce small file issues.
- **Adaptive Query Execution**: Enabled for dynamic resource allocation and optimization during query execution.

---

### **8.2 Observations from Spark UI**

To ensure optimal performance, the following five dimensions were reviewed using **Spark UI metrics**:

<img width="1728" alt="Executors_1" src="https://github.com/user-attachments/assets/55a96d1b-45ed-4f6c-b265-e479eba18c36" />

<img width="1728" alt="Executors_2" src="https://github.com/user-attachments/assets/ecb86946-9b10-4807-89af-1d74f1f265bb" />

<img width="1728" alt="Stages" src="https://github.com/user-attachments/assets/21dd3fd9-59d5-45ab-a99e-68787c7cd05d" />

<img width="1728" alt="Storage" src="https://github.com/user-attachments/assets/217eec05-541d-436b-8f9a-4c09a1e5a8cc" />

---

#### **1. Spill (Memory vs Disk Writes)**
- **Observation**: No memory spill (both on-heap and off-heap memory usage stayed within limits).
- **Interpretation**: This indicates the executor memory allocation was sufficient for the task, showing proper memory management and configuration.
- **Optimization Strategy**: 
  - Carefully tuned memory-related configurations (`spark.sql.shuffle.partitions`, `spark.memory.fraction`) prevented spillovers.
  - Spark's default settings for task execution were reviewed, and memory utilization remained within safe thresholds.

---

#### **2. Skew (Partition Size Imbalance)**
- **Observation**: Task distribution across executors was even (e.g., **8023** and **8026** tasks completed across executors).
- **Interpretation**: Minimal data skew ensured uniform load distribution, avoiding delays caused by outlier partitions.
- **Potential Enhancements**:
  - Introduce **salting** or **custom partitioning** if data volume increases significantly to preempt any future skew.
  - Implement **`repartition()`** in areas where future data patterns may create imbalanced partitions.

---

#### **3. Shuffle (Network I/O)**
- **Observation**: Shuffle read/write size was **33.7 MiB**—relatively small for wide transformations.
- **Interpretation**: Efficiently structured transformations (e.g., `groupBy` and `join`) minimized unnecessary shuffles and network I/O.
- **Optimization Techniques Used**:
  - **Broadcast Joins**: Applied where feasible to avoid full shuffle operations.
  - Combined transformations (e.g., aggregations) to reduce the number of stages in the query execution.

---

#### **4. Storage (Disk Usage and Small File Issues)**
- **Observation**: Total disk space usage was **463.9 GiB**; Parquet cache hit ratio was **62%**, indicating effective caching.
- **Interpretation**: The cache helped avoid repeated disk reads by storing frequently accessed data.
- **Enhancements Applied**:
  - Delta Lake optimizations (`optimize`, `z-ordering`, `compaction`) were employed to manage storage layout efficiently and avoid performance penalties due to small files.
  - Caching frequently queried datasets to improve query speed and minimize disk I/O.

---

#### **5. Serialization (Data Serialization and Deserialization)**
- **Observation**: No significant serialization/deserialization overhead observed.
- **Interpretation**: Efficient serialization formats (Parquet) and Spark's internal Tungsten execution engine reduced serialization costs.
- **Future Considerations**:
  - Enable **Kryo Serialization** to further improve performance for large data objects, as Kryo is faster and more space-efficient than Java serialization.
  - Evaluate the use of **object pooling** to reduce serialization time in more resource-intensive workloads.

---

### **8.3 Overall Impact**
These observations underscore that the pipeline was well-configured to handle the streaming workload efficiently:
- **No evidence** of performance bottlenecks due to memory spills or serialization issues.
- **Uniform partitioning** ensured balanced parallel processing across all executors.
- **Minimal shuffle data movement**, coupled with efficient storage strategies, reduced I/O latency.

---

### **Next Steps for Further Optimization:**
1. **Dynamic Allocation Tuning**: Adjust executor allocation based on fluctuating workloads to optimize resource usage further.
2. **Experiment with Caching**: Increase Parquet cache size to improve query speeds for frequent lookups.
3. **Scalability Testing**: Perform stress tests with larger datasets to validate scalability and resilience under increased loads.

---

Overall, this highlights that the streaming pipeline was both performant and scalable, with optimizations aligned with Spark's best practices for memory usage, partitioning, shuffle handling, and efficient storage formats.

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
