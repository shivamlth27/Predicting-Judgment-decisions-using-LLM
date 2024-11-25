Here’s a **detailed README.md** for your project, incorporating more specifics from your documents and adding depth to each section. It uses headers, examples, and elaborates on methodology and significance.

---

# Predicting Judgment Decisions Using LLMs

This repository is part of the **DS504: Natural Language Processing** project by Group 17, which explores leveraging state-of-the-art **Large Language Models (LLMs)** to predict judicial outcomes and provide explainable rationales. The project introduces **PredEx**, the largest annotated dataset of Indian legal judgments, and implements advanced NLP techniques to address critical challenges in the judiciary.

---

## 📝 **Project Summary**

Legal systems worldwide face the daunting challenge of addressing **millions of pending cases**. In India alone, over **30 million cases** are backlogged, with predictions of centuries to clear them unless immediate interventions are implemented. 

This project leverages **Natural Language Processing (NLP)** to:

1. **Predict judicial outcomes** for cases based on facts.
2. **Explain predictions** through interpretable AI techniques, aiding transparency.
3. **Enable scalability** for legal systems globally by using domain-specific fine-tuned models.

By introducing **PredEx**, an expert-annotated dataset tailored to Indian judicial needs, and employing **fine-tuned LLMs** like RoBERTa and Legal-BERT, this project sets a benchmark for **legal NLP applications**.

---

## 🌟 **Key Features**

### 1. PredEx Dataset
- **Largest dataset** for Indian legal judgment prediction and explainability.
- **Sources**: Indian Supreme Court and High Court judgments curated via **IndianKanoon**.
- **Annotations**: Expert-labeled key sentences crucial for decisions.

| **Metric**            | **Training Set** | **Testing Set** |
|-----------------------|-----------------|----------------|
| **Documents**         | 12,178         | 3,044          |
| **Avg Tokens/Doc**    | 4,586          | 4,422          |
| **Max Tokens**        | 117,733        | 83,657         |

### 2. Models
- Fine-tuned LLMs (**RoBERTa, Legal-BERT**) trained with **instruction-tuning** for domain-specific tasks.
- Hierarchical models process long legal texts, aggregating **paragraph-level embeddings** for comprehensive predictions.

### 3. Explainability
- Attention-based mechanisms provide **justifications** for predictions.
- Evaluated using a **Likert scale** by legal experts for transparency.

### 4. Results and Impact
- **Macro F1 Score**: Achieved **83.4%** with advanced architectures.
- Enhanced transparency and **efficiency in judicial decision-making**.

---

## 📂 **Repository Structure**

```plaintext
├── data/
│   ├── predex_train.json   # Training dataset (80%)
│   ├── predex_test.json    # Testing dataset (20%)
├── models/
│   ├── fine_tuned_roberta/ # Checkpoints of fine-tuned RoBERTa
│   ├── fine_tuned_legalbert/ # Legal-BERT model
├── src/
│   ├── preprocessing.py    # Data cleaning and preparation
│   ├── chunking.py         # Code for managing long legal texts
│   ├── training.py         # Model training script
│   ├── evaluation.py       # Model evaluation script
├── notebooks/
│   ├── eda.ipynb           # Exploratory Data Analysis
│   ├── explainability.ipynb # Attention visualization
├── results/
│   ├── performance_metrics.csv # Detailed model metrics
│   ├── attention_visualizations/ # Attention score outputs
├── assets/
│   ├── model_architecture.png  # Architecture diagram
│   ├── attention_example.png   # Sample attention visualization
├── README.md               # Project documentation
└── requirements.txt        # Dependencies
```

---

## ⚙️ **Methodology**

1. **Problem Definition**
   - **Goal**: Predict judicial outcomes (binary/multi-class) and provide **human-interpretable justifications**.
   - **Challenges**:
     - Long documents with up to 100,000 tokens.
     - Imbalanced dataset for specific human rights violations.

2. **Dataset Compilation**
   - **Sources**: Legal cases from **IndianKanoon**.
   - **Annotation**:
     - Legal experts highlighted sentences crucial for decisions.
     - Sentences annotated for both **outcome prediction** and **relevance explanation**.

3. **Model Training**
   - Fine-tuned **RoBERTa** and **Legal-BERT** on PredEx.
   - Chunked documents into manageable sections for processing.
   - Hierarchical models aggregated paragraph embeddings for case-level predictions.

4. **Explainability**
   - Introduced **attention visualization** for paragraph relevance.
   - Used **Likert-scale expert evaluations** for explanation quality.

---

## 🚀 **Results**

### Binary Classification Results

| **Model**         | **Precision** | **Recall** | **Macro F1** |
|-------------------|--------------|-----------|-------------|
| **RoBERTa**       | 85.7%       | 93.9%     | 83.3%       |
| **Legal-BERT**    | 86.3%       | 90.0%     | 81.8%       |
| **HIER-BERT**     | 91.3%       | 80.5%     | **83.3%**   |
| **aLEXa-BERT**    | **91.1%**   | **80.6%** | **83.4%**   |

### Explainability Example

Below is a sample **attention heatmap** for the case *Maxian And Maxianová v. Slovakia (2014)*:

**Predicted Outcome**: Violation of Article 6 - Right to a Fair Trial  
![Attention Heatmap](assets/attention_example.png)

---

## 📈 **Future Directions**

1. **Domain-Specific LLMs**:
   - Fine-tune models for other areas like criminal or corporate law.
   
2. **Bias Mitigation**:
   - Implement named-entity anonymization (e.g., masking names, locations).
   
3. **Scalability**:
   - Apply the methodology to other judicial systems globally.

4. **Reinforcement Learning**:
   - Use **reinforcement learning from human feedback (RLHF)** for improved explanations.

---

## ⚡ **How to Get Started**

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Libraries from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

### Steps to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/judgment-llm.git
   cd judgment-llm
   ```

2. Prepare the dataset:
   - Place the **PredEx dataset** files in the `data/` directory.

3. Train a model:
   ```bash
   python src/training.py --model roberta --dataset data/predex_train.json
   ```

4. Evaluate the model:
   ```bash
   python src/evaluation.py --model fine_tuned_roberta
   ```

5. Visualize results:
   - Use the `notebooks/explainability.ipynb` notebook for attention heatmaps.

---

## 🤝 **Acknowledgments**

This project was developed as part of **DS504: Natural Language Processing** at IIT Bhilai by:

- **Aditya Prakash**
- **Akshat Kumar**
- **Ayush Kumar Mishra**
- **Shivam**

Special thanks to:
- The **IndianKanoon** team for providing legal case data.
- Mentors and domain experts for guidance.

---

## 📚 **References**

1. Chalkidis et al., "Neural Legal Judgment Prediction in English," 2019.
2. Nallapati & Manning, "Legal Docket Classification," 2008.
3. [spaCy Library for NLP](https://spacy.io)

---

Feel free to suggest additions or provide example images for placeholders!
