# Importações de bibliotecas necessárias
import os
from langchain.agents import Tool  # Define ferramentas personalizadas para o agente usar
from langchain.prompts import PromptTemplate  # Cria templates de prompt
from langchain_experimental.utilities import PythonREPL  # Shell Python para execução de código
from langchain_experimental.agents.agent_toolkits import create_python_agent  # Cria agente com suporte Python
from langchain_openai import ChatOpenAI  # Integração com modelos da OpenAI

# Define a chave da API da OpenAI
os.environ['OPENAI_API_KEY'] = 'SUA CHAVE DE API'  # Substitua por sua chave real ou use dotenv para segurança

# Criação do modelo LLM com GPT-3.5 Turbo
model = ChatOpenAI(model='gpt-3.5-turbo')

# Instancia o shell Python para execução de código em tempo real
python_repl = PythonREPL()

# Cria uma ferramenta que descreve o uso do shell Python para o agente
python_repl_tool = Tool(
    name='Python REPL',
    description='Um shell Python. Use isso para executar código Python. '
                'Execute apenas códigos Python válidos. '
                'Se você precisar obter o retorno do código, use a função "print(...)".',
    func=python_repl.run
)

# Cria o agente, conectando o modelo LLM com a ferramenta definida
agent_executor = create_python_agent(
    llm=model,
    tool=python_repl_tool,
    verbose=True  # Mostra logs detalhados da execução
)

# Criação de um template de prompt simples
prompt_template = PromptTemplate(
    input_variables=['query'],
    template='Resolva o problema: {query}.'
)

# Definição da pergunta
query = r'quanto é 20% de 3000'

# Formata o prompt com a pergunta
prompt = prompt_template.format(query=query)

# Executa o agente com o prompt
response = agent_executor.invoke(prompt)

# Exibe a resposta retornada pelo agente
print(response.get('output'))
