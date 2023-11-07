import streamlit as app_interface
import os
import pickle
from dotenv import dotenv_values
from PyPDF2 import PdfFileReader
from streamlit_extras import space_filler
from langchain.text_processors import TextChunker
from langchain.embeddings import GPTEmbedder
from langchain.indexes import FAISSIndex
from langchain.models import GPTModel
from langchain.pipelines import qa_pipeline_loader
from langchain.hooks import openai_interaction_hook

# Configuration load
config = dotenv_values()

# Define the main interface
def interactive_chatbot():
    app_interface.header("Interactive PDF Dialogue 📘💭")

    # PDF upload section
    uploaded_pdf = app_interface.file_uploader("Choose a PDF document", type=['pdf'])

    if uploaded_pdf:
        pdf_handler = PdfFileReader(uploaded_pdf)
        document_text = ""
        for page in pdf_handler.pages:
            document_text += page.extractText()

        chunk_processor = TextChunker(
            max_size=1000,
            overlap_size=200,
            size_getter=len
        )
        document_chunks = chunk_processor.divide_text(document_text)

        # Prepare embeddings store
        pdf_base_name = os.path.splitext(uploaded_pdf.name)[0]
        embeddings_file = f"{pdf_base_name}.pkl"

        if os.path.isfile(embeddings_file):
            with open(embeddings_file, "rb") as file:
                text_index = pickle.load(file)
        else:
            embedder = GPTEmbedder()
            text_index = FAISSIndex.create_from_documents(document_chunks, embedder)
            with open(embeddings_file, "wb") as file:
                pickle.dump(text_index, file)

        # Question input
        user_question = app_interface.text_input("Enter your question about the PDF content:")

        if user_question:
            related_documents = text_index.search_similar_documents(user_question, num_results=3)

            large_language_model = GPTModel()
            question_answering_chain = qa_pipeline_loader(model=large_language_model, pipeline_type="basic")
            with openai_interaction_hook() as interaction:
                answer = question_answering_chain.execute(docs=related_documents, query=user_question)
            app_interface.write(answer)

# Interface sidebar
with app_interface.sidebar:
    app_interface.image('logo.png', width=150)  # Display logo
    app_interface.title('🤖📚 PDF Bot')
    app_interface.markdown('''
    ### Info
    This tool is built with:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://openai.com/) models

    ''')
    space_filler(5)
    app_interface.caption('Crafted with 🧡 by Your Company Name')

# Run the app
if __name__ == "__main__":
    interactive_chatbot()
