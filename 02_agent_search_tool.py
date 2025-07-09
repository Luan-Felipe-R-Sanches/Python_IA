import os
from langchain.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.agents.agent_toolkits import create_python_agent  

from langchain_community.chat_models import ChatOpenAI 

# Define a chave da API do OpenAI
os.environ['OPENAI_API_KEY'] = 'KEY'  # Substitua 'KEY' pela sua chave real

# Instancia o modelo OpenAI
model = ChatOpenAI(model='gpt-3.5-turbo')

# Cria a ferramenta de busca na Wikipédia em português
wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        lang='pt'
    )
)

# Cria o agente com a ferramenta do Python e a LLM (Large Language Model)
agent_executor = create_python_agent(
    llm=model,
    tools=[wikipedia_tool],  # Corrigido: o argumento espera uma lista de ferramentas
    verbose=True,
)

# Define o template do prompt para realizar a pesquisa
prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
    Pesquise na web sobre {query} e forneça um resumo sobre o assunto.
    Responda tudo em português brasileiro.
    '''
)

# Define a consulta
query = 'Alan Turing'
prompt = prompt_template.format(query=query) 
# Executa o agente com o prompt
response = agent_executor.invoke({'input': prompt}) 

# Imprime a resposta
print(response.get('output'))
