import os
import streamlit as st
import pickle
import time
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

st.title('Chat with a Webapp')
st.sidebar.title('Provide the link to the website')

# Process the query and display the answer to the user
def process_query(query, llm):
    if query and 'vectorIndex' in st.session_state:
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=st.session_state['vectorIndex'].as_retriever())
        result = chain.invoke({"question": query}, return_only_outputs=True)
        st.header("Answer")
        st.write(result["answer"])
    else:
        if query:
            st.header("Answer")
            st.write("Please process a URL first.")

# Process the URL and create vector embeddings
def process_url(url, main_placeholder):
    try:
        loaders = UnstructuredURLLoader(urls=[url])
        data = loaders.load()

        if data:
            # Split the text into smaller chunks
            splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', ' '],
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(data)

            # Embed the chunks using BAAI/bge-small-en model
            model_name = "BAAI/bge-small-en"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )

            vectorstore = FAISS.from_documents(chunks, embeddings)

            # Serialize the vectorstore object
            file_path = "vectordatabase.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(vectorstore, f)

            # Load the vectorstore into session state
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    st.session_state['vectorIndex'] = pickle.load(f)

            main_placeholder.text("Start Chatting...✅✅✅")
            time.sleep(1)
            return True
        else:
            main_placeholder.text("Error processing the URL, please provide a valid URL.")
            return False

    except Exception as e:
        main_placeholder.text(f"Error processing the URL: {e}")
        return False

# Main function to load the model and handle user input
def main():
    # Load the model from the HuggingFace Hub
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        model_kwargs={'max_length': 256, 'token': os.environ['HUGGINGFACEHUB_API_TOKEN']},
        temperature=0.3,
    )

    url = st.sidebar.text_input('Enter the URL')
    process_url_clicked = st.sidebar.button("Click to process the Article")

    main_placeholder = st.empty()

    # Process the URL if the button is clicked
    if process_url_clicked:
        if url:
            success = process_url(url, main_placeholder)
            if success:
                st.session_state['url_processed'] = True
        else:
            main_placeholder.text("Please enter a valid URL.")

    # Display the question input only if the URL has been successfully processed
    if st.session_state.get('url_processed'):
        query = st.text_input("Question: ")
        if query:
            process_query(query, llm)

if __name__ == "__main__":
    main()