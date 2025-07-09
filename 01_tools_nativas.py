from langchain_community.tools import DuckDuckGoSearchRun

ddg_search = DuckDuckGoSearchRun()

search_result = ddg_search.run('Quan foi Alan Turing?')
print(search_result)