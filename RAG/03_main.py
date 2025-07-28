import os
from langchain.chains.retriever import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "KEY"

model = ChatOpenAI(
    model='gpt-4',
)

persist_directory = 'db'
embedding=OpenAIEmbeddings()

vector_store=Chroma(
    persist_directory=persist_directory,
    embedding_fuction=embedding,
    collection_name='proposta'
)

retriever = vector_store.as_retriever()

system_prompt = '''
Use o contexto para responder as perguntas.
Contexto: {context}
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt)
        ('human', {input})
    ]
)
question_answer_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
)

chain =  create_retrieval_chain()