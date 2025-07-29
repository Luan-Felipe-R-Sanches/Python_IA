import os
from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Defina sua chave da API OpenAI
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Caminho do arquivo CSV
csv_path = "carros.csv"

# Carrega o CSV como documentos (cada linha/registro vira um documento)
loader = CSVLoader(csv_path)
docs = loader.load()

# Divide os documentos em pedaços menores para melhorar indexação
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents=docs)

# Diretório para persistir os vetores
persist_directory = "db"

# Inicializa o modelo de embeddings da OpenAI
embedding = OpenAIEmbeddings()

# Cria ou carrega o banco vetorial Chroma com os documentos indexados
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name="carros",
)
