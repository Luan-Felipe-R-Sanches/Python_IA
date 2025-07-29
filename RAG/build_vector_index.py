import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Defina sua chave da API OpenAI
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Caminho do arquivo PDF
pdf_path = "Proposta.pdf"

# Carrega o PDF em documentos (cada página como documento)
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Divide os documentos em pedaços menores para indexação
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents=docs)

# Diretório onde o vetor será persistido
persist_directory = "db"

# Inicializa o gerador de embeddings
embedding = OpenAIEmbeddings()

# Cria ou carrega o banco vetorial a partir dos documentos divididos
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name="proposta",
)
