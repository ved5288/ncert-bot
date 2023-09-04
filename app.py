import streamlit as st
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

from gsheet import push_messages_to_sheet
 
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 NCERT X class Economics bot')
    st.markdown('''
    ## About
    This app is a smart question answer bot on the X class NCERT Economics syllabus
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Vedant Somani](https://www.linkedin.com/in/vedantsomani/)')
 
def get_response_for_query(vector_store, query):
    docs = vector_store.similarity_search(query=query, k=3)

    print(docs)
 
    llm = OpenAI(openai_api_key = st.secrets["OPENAI_API_KEY"], model_name='gpt-3.5-turbo')
    # llm = OpenAI()
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        return response
    return "Something went wrong while processing your request. Please contact the author."


def read_pdf(pdf_file_path):
    pdf = open(pdf_file_path, "rb")
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        page_text = page_text.replace('\n', ' ')
        text += page_text
    
    # pdf.close()
    return text

def create_dataset(directory, chunk_size, chunk_overlap=50):
    files_in_dir = [directory + f for f in os.listdir(directory) if f[-4:] == ".pdf"]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = []
    for file_ in files_in_dir:
        print("Processing " + file_)
        pdf_text = read_pdf(file_)
        file_chunks = text_splitter.split_text(text=pdf_text)
        chunks += file_chunks

    return chunks


def main(chunk_size):
    st.header("Understanding Economic Development 💬")
    if not os.path.exists(f"ncert_{chunk_size}.pkl"):
        chunks = create_dataset("./eco-x/", chunk_size)
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"ncert_{chunk_size}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
    else:
        with open(f"ncert_{chunk_size}.pkl", "rb") as f:
            # Embeddings loaded from disk.
            VectorStore = pickle.load(f)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = get_response_for_query(VectorStore, prompt)
        push_messages_to_sheet(prompt, response)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
 
if __name__ == '__main__':
    main(600)


