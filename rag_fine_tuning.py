import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import torch
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LegalRAGSystem:
    def __init__(self):
        """
        Initialize the Legal RAG System
        """
        # Verify HuggingFace API token is set
        self.hf_token = "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"
        if not self.hf_token:
            raise ValueError("Please set HUGGINGFACE_API_TOKEN environment variable")
            
        self.embeddings = None
        self.vectorstore = None
        self.model = None
        
    def prepare_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare the dataset for RAG by combining relevant fields
        """
        documents = []
        for _, row in df.iterrows():
            document = {
                'content': f"""Case: {row['Case Name']}
                Input: {row['Input']}
                Output: {row['Output']}
                Instruction: {row['text']}
                Label: {row['Label']}""",
                'metadata': {
                    'case_name': row['Case Name'],
                    'label': row['Label'],
                    'instruction': row['text']
                }
            }
            documents.append(document)
        return documents
    
    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Split documents into smaller chunks for better retrieval
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        splits = []
        for doc in documents:
            chunks = text_splitter.split_text(doc['content'])
            for chunk in chunks:
                splits.append({
                    'content': chunk,
                    'metadata': doc['metadata']
                })
        return splits
    
    def setup_embeddings(self):
        """
        Initialize the embedding model
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    def create_vectorstore(self, documents: List[Dict]):
        """
        Create FAISS vectorstore from documents
        """
        texts = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
    
    def setup_retrieval_chain(self):
        """
        Setup the retrieval chain with custom prompt
        """
        prompt_template = """
        You are a legal assistant. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Initialize the language model with the new HuggingFaceEndpoint
        self.model = HuggingFaceEndpoint(
            repo_id="google/flan-t5-large",
            huggingfacehub_api_token=self.hf_token,
            temperature=0.7,
            max_length=512
        )
        
        # Create the retrieval chain
        self.chain = RetrievalQA.from_chain_type(
            llm=self.model,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def save_vectorstore(self, path: str):
        """
        Save the FAISS vectorstore
        """
        self.vectorstore.save_local(path)
    
    def load_vectorstore(self, path: str):
        """
        Load the FAISS vectorstore
        """
        self.vectorstore = FAISS.load_local(path, self.embeddings)

def main():
    # Initialize the RAG system
    rag_system = LegalRAGSystem()
    
    # Load your dataset
    df = pd.read_csv("datasets/train_ft.csv")
    df= df.head(100)
    
    # Prepare documents
    print("Preparing documents...")
    documents = rag_system.prepare_data(df)
    
    # Split documents
    print("Splitting documents...")
    splits = rag_system.split_documents(documents)
    
    # Setup embeddings
    print("Setting up embeddings...")
    rag_system.setup_embeddings()
    
    # Create vectorstore
    print("Creating vector store...")
    rag_system.create_vectorstore(splits)
    
    # Save vectorstore
    print("Saving vector store...")
    rag_system.save_vectorstore("legal_vectorstore")
    
    # Setup retrieval chain
    print("Setting up retrieval chain...")
    rag_system.setup_retrieval_chain()
    
    # Example query
    query = "What was the verdict in the KAMLESH Vs. UNION OF INDIA case?"
    result = rag_system.chain(query)
    print("Query result:", result['result'])

if __name__ == "__main__":
    main()