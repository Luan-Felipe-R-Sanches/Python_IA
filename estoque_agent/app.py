from decouple import config

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

st.set_page_config(
    page_title="Estoque GPT",
    page_icon="📄",
)

st.header("Assistente de Estoque")

model_options = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o-mini",
    "gpt-4o",
]

selected_model = st.sidebar.selectbox(
    label="Selecione o modelo LLM",
    options=model_options,
)

st.sidebar.markdown("### Sobre")
st.sidebar.markdown(
    "Este agente consulta um banco de dados de estoque utilizando um modelo GPT."
)

st.write("Faça perguntas sobre o estoque de produtos, preços e reposições.")

user_question = st.text_input("O que deseja saber sobre o estoque?")

model = ChatOpenAI(
    model=selected_model,
)
db = SQLDatabase.from_uri('sqlite://estoque.db')
toolkit = SQLDatabaseToolkit
if st.button('Consultar'):
    if user_question:
        st.write('FEZ UMA PERGUNTA!')
    else:
        st.warning('Por favor, insira uma pergunta.')
