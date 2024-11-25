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
├── images/                   # Contains visual assets like architecture diagrams and results
├── templates/                # Contains HTML templates for the web app (if applicable)
├── .gitignore                # Files and folders to exclude from Git versioning
├── README.md                 # Project documentation
├── app.py                    # Backend API or web application for interfacing with the model
├── data_prep.ipynb           # Data preprocessing scripts and notebook
├── inference.py              # Model inference scripts for testing predictions
├── pretrained_llama.py       # Pre-trained model implementation (LLama architecture)
├── report.pdf                # Detailed project report
├── roberta.py                # RoBERTa fine-tuning implementation
├── roberta_train.py          # Script for training the RoBERTa model
├── train.ipynb               # Training pipeline notebook
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

![Validation](https://github.com/user-attachments/assets/dfe00b8f-df17-4f43-8890-6249ebbf6206)


![WhatsApp Image 2024-11-25 at 10 40 34_7348b125](https://github.com/user-attachments/assets/20a57296-71b6-42c6-ac23-0c6cf8b30d03)
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
   git clone https://github.com/shivamlth27/Predicting-Judgment-decisions-using-LLM.git
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

---

## 📚 **References**

1. Here's how you can include the link to your paper in the README:

1. **[From RoBERTa to aLEXa: Automated Legal Expert Arbitrator for Neural Legal Judgment Prediction](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1224/reports/custom_116826881.pdf)**

2. **[Legal Judgment Reimagined: PredEx and the Rise of Intelligent AI Interpretation in Indian Courts](https://arxiv.org/abs/2406.04136)**  
   

