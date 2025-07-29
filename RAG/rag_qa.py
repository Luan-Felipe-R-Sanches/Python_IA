import os
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Defina sua chave da API OpenAI
os.environ["OPENAI_API_KEY"] = "SUA_CHAVE_AQUI"

#  Inicializa o modelo de linguagem
model = ChatOpenAI(
    model="gpt-4",
)

# Diretório onde os vetores serão armazenados
persist_directory = "db"

#  Cria o embedding para transformar texto em vetores
embedding = OpenAIEmbeddings()

#  Inicializa ou carrega o banco vetorial
vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
    collection_name="proposta",
)

# 🔍 Cria um retriever para buscar os documentos relevantes
retriever = vector_store.as_retriever()

#  Prompt de sistema que dá instruções ao modelo
system_prompt = """
Use o contexto abaixo para responder com precisão.
Contexto: {context}
"""

#  Prompt final com mensagem do usuário
prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)

#  Cadeia que combina documentos com o modelo
question_answer_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
)

#  Cadeia de recuperação + resposta
chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

#  Pergunta feita ao sistema
query = "Qual é a proposta do documento?"

#  Executa a cadeia
response = chain.invoke({"input": query})

#  Exibe a resposta
print(response["answer"])
