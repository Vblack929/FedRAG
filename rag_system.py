from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Sized,
    Tuple,
    Union,
)
from pathlib import Path
import os
import json
import random
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from contriever.src.contriever import Contriever 
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from abc import ABC, abstractmethod
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from llm import GPT
os.environ['KMP_DUPLICATE_LIB_OK']='True'

load_dotenv()

class ContrieverEmbeddings(Embeddings):
    """Contriever embeddings wrapper for use with LangChain."""
    
    def __init__(self, model_name="facebook/contriever"):
        """Initialize the Contriever model.
        
        Args:
            model_name: Hugging Face model ID for Contriever
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Contriever.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Contriever."""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        embeddings = self.model(**inputs)
        
        return embeddings.detach().cpu().numpy()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text using Contriever."""
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        ).to(self.device)
        
        embeddings = self.model(**inputs)
        return embeddings.detach().cpu().numpy()
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling of token embeddings weighted by attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SimpleRAG:
    """A simple RAG implementation focusing on core embedding and similarity functionality."""
    
    def __init__(
        self,
        embedding_function: Union[Callable[[str], List[float]], Embeddings],
        normalize: bool = True
    ):
        """Initialize the RAG system.
        
        Args:
            embedding_function: Either a callable that takes a string and returns a list of floats,
                              or an Embeddings object from langchain
            normalize: Whether to normalize embeddings to unit L2 norm
        """
        self.embedding_function = embedding_function
        self.normalize = normalize
        self.documents: List[Document] = []
        self.embeddings: List[List[float]] = []
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        if isinstance(self.embedding_function, Embeddings):
            embedding = self.embedding_function.embed_query(text)
        else:
            embedding = self.embedding_function(text)
            
        if self.normalize:
            embedding = self._normalize(embedding)
            
        return embedding
    
    def _normalize(self, embedding: List[float]) -> List[float]:
        """Normalize embedding to unit L2 norm."""
        embedding = np.array(embedding)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the RAG system."""
        if not documents:
            return
            
        # Get embeddings for all documents
        texts = [doc.page_content for doc in documents]
        if isinstance(self.embedding_function, Embeddings):
            embeddings = self.embedding_function.embed_documents(texts)
        else:
            embeddings = [self.embedding_function(text) for text in texts]
            
        if self.normalize:
            embeddings = [self._normalize(emb) for emb in embeddings]
            
        # Store documents and their embeddings
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return documents most similar to query."""
        if not self.documents:
            return []
            
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        if isinstance(self.embedding_function, ContrieverEmbeddings):
            similarities = []
            for doc_embedding in self.embeddings:
                similarities.append((query_embedding @ doc_embedding).item())
        else:
            similarities = [
                np.dot(query_embedding, doc_embedding)
                for doc_embedding in self.embeddings
            ]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return documents
        return [self.documents[i] for i in top_k_indices]
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Return documents most similar to query and score for each."""
        if not self.documents:
            return []
            
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Check if we're using Contriever embeddings
        if isinstance(self.embedding_function, ContrieverEmbeddings):
            # Convert to numpy arrays for matrix multiplication
            
            # Calculate similarities using matrix multiplication
            similarities = []
            for doc_embedding in self.embeddings:
                similarities.append((query_embedding @ doc_embedding).item())
            
            # Get top k indices and scores
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_scores = [similarities[i] for i in top_k_indices]
            
        else:
            # Calculate similarities using dot product
            similarities = [
                np.dot(query_embedding, doc_embedding)
                for doc_embedding in self.embeddings
            ]
            
            # Get top k indices and scores
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_scores = [similarities[i] for i in top_k_indices]
        
        # Return documents with scores
        return [(self.documents[i], float(score)) 
                for i, score in zip(top_k_indices, top_k_scores)]
    
    def clear(self) -> None:
        """Clear all documents and embeddings."""
        self.documents = []
        self.embeddings = []

def main():
    data_path = "data/poisoned/nq/nq_long_answer_attacks.json"
    
    # Print header
    print("=== Testing Contriever Embeddings with 5 Samples ===")
    
    # Load the data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Take only the first 5 samples for testing
    samples = data[:5]
    
    print("Sample questions:")
    for i, sample in enumerate(samples):
        print(f"{i+1}. {sample['question']}")
    
    print("\n=== Initializing Contriever Embeddings ===")
    embedding_function = ContrieverEmbeddings()
    
    # Create SimpleRAG instance
    rag = SimpleRAG(
        embedding_function=embedding_function,
        normalize=False
    )
    
    # Prepare documents
    documents = []
    for i, sample in enumerate(samples):
        # Add long answer as document
        doc = Document(
            page_content=sample['long_answer'],
            metadata={
                'sample_id': f"sample_{i}",
                'question': sample['question']
            }
        )
        documents.append(doc)
    
    # Add documents to RAG
    print(f"\nAdding {len(documents)} documents to RAG system")
    rag.add_documents(documents)
    
    # Test each question
    print("\n=== Testing Retrieval for Each Question ===")
    for i, sample in enumerate(samples):
        query = sample['question']
        print(f"\nQuery {i+1}: {query}")
        
        # Get top 3 results
        docs_with_scores = rag.similarity_search_with_score(query, k=3)
        
        print("Top 3 retrieved documents:")
        for j, (doc, score) in enumerate(docs_with_scores):
            sample_id = doc.metadata.get('sample_id', 'unknown')
            question = doc.metadata.get('question', 'unknown')
            print(f"  {j+1}. Score: {score:.4f}, Sample ID: {sample_id}")
            print(f"     Original question: {question}")
            print(f"     Content (first 100 chars): {doc.page_content[:100]}...")
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    main()