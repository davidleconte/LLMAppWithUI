

# This script is a Streamlit application that uses the Hugging Face transformers library
# to create a chatbot that can answer questions based on the content of an HTML page.


import os
import pickle
import requests
import streamlit as st
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi
import faiss

class App:
    _model: AutoModelForCausalLM
    _tokenizer: AutoTokenizer

    def __init__(self) -> None:
        # Initialize Hugging Face API with the provided API key
        hf_api = HfApi(token="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")
        # Load the tokenizer and model from the Hugging Face model hub
        self._tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        self._model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        self.prompt = """What is {txt}?"""
        # Configurable file path for the FAISS index
        self.file_path = "db/faiss_store.pkl"

    def _ensure_directory_exists(self, path: str):
        """Ensure the directory exists. If not, create it."""
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _get_html_text(self, url):
        # Fetch the HTML content from the given URL and extract the text using BeautifulSoup
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        txt = soup.get_text()
        lines = (line.strip() for line in txt.splitlines())
        return "\n".join(line for line in lines if line)

    def embed_documents(self, texts):
        # Generate embeddings for a list of texts using the Hugging Face model
        inputs = self._tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Use the mean of the last hidden state as the embedding
        return embeddings.detach().numpy()

    def __call__(self):
        # Create the Streamlit app interface
        st.write("# Ask html page")

        # Input field for the URL
        url = st.text_input("URL", "https://arxiv.org/abs/2308.14963")
        if url:
            # Fetch the text from the URL and generate embeddings
            txt = self._get_html_text(url)
            embeddings = self.embed_documents([txt])
            # Create a FAISS index from the embeddings and save it to a file
            faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
            faiss_index.add(embeddings)
            self._ensure_directory_exists(self.file_path)
            with open(self.file_path, "wb") as f:
                pickle.dump((faiss_index, [{"source": url}]), f)

            # Load the FAISS index from the file
            with open(self.file_path, "rb") as f:
                faiss_index, metadatas = pickle.load(f)

            # Input field for the user's question
            q = st.text_input("What is ...?")
            if q is None or q == "":
                return

            # Generate embeddings for the user's question and search the FAISS index for the closest document
            question_embedding = self.embed_documents([q])
            _, closest_document_idx = faiss_index.search(question_embedding, 1)
            closest_document_metadata = metadatas[closest_document_idx[0][0]]

            # Display the source URL of the closest document as the answer
            st.write(f"Answer based on the document at {closest_document_metadata['source']}.")


if __name__ == "__main__":
    app = App()
    app()