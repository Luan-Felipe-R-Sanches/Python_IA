
import os
from langchain import hub
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI

# Configure a chave da API com segurança
os.environ['OPENAI_API_KEY'] = 'KEY'  # Substitua 'KEY' pela sua chave real

# Criação do modelo LLM com GPT-3.5 Turbo
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# Prompt principal em português, com variável de entrada 'q'
prompt = '''
Você é um assistente financeiro pessoal, que responderá as perguntas dando dicas financeiras e de investimento.
Responda tudo em português brasileiro.
Pergunta: {q}
'''
prompt_template = PromptTemplate.from_template(prompt)

# Ferramenta para execução de código Python (cálculos financeiros)
python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description=(
        'Um Shell Python. Use isso para executar código Python. Execute apenas códigos Python válidos. '
        'Se você precisar obter o retorno do código, use a função "print(...)". '
        'Use para realizar cálculos financeiros necessários para responder as perguntas e dar dicas.'
    ),
    func=python_repl.run,
)

# Ferramenta de busca na internet (DuckDuckGo)
search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name='Busca DuckDuckGo',
    description=(
        'Útil para encontrar informações e dicas de economia e opções de investimento. '
        'Você sempre deve pesquisar na internet as melhores dicas usando esta ferramenta. '
        'Responda informando que as informações foram pesquisadas na internet.'
    ),
    func=search.run,
)

# Carregando o prompt ReAct da LangChain Hub (modelo de tomada de decisão)
react_instructions = hub.pull('hwchase17/react')  # Esse modelo espera que o prompt seja passado dinamicamente

# Lista de ferramentas disponíveis para o agente
tools = [python_repl_tool, duckduckgo_tool]

# Criação do agente com a LLM, as ferramentas e o prompt base
agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_instructions
)

# Executor do agente (responsável por fazer o processo interativo com o usuário)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Mostra os passos de raciocínio do agente no terminal
)

# Exemplo de pergunta do usuário
question = '''
Minha renda é de R$10000 por mês, o total de minhas despesas é de R$8500 mais 1000 de aluguel.
Quais dicas de investimento você me dá?
'''

# Invoca o agente com a entrada formatada
output = agent_executor.invoke(
    {'input': prompt_template.format(q=question)}  # Usa o prompt personalizado com a pergunta embutida
)

# Exibe a resposta final do agente
print(output.get('output'))
