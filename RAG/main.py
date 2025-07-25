import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import hub  # ✅ Import this

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Load PDF
pdf_path = "Proposta.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(docs)

# Create vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="laptop_manual",
)

# Set up retriever
retriever = vector_store.as_retriever()

# Load prompt
prompt = hub.pull("rlm/rag-prompt")

# Set up model
model = ChatOpenAI(model="gpt-4")

# Assemble RAG chain
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)

try:
    while True:
        question = input("Qual a sua dúvida? ")
        response = rag_chain.invoke(question)
        print(response)

except KeyboardInterrupt:
    exit()
