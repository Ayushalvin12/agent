import json
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_txt(data: str, filename: str = "research_output.txt"):
    if not isinstance(data, str):
        try:
            data = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            data = str(data)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name = "save_text_to_file",
    func=save_to_txt,
    description="Saves the research output to a text file with a timestamp. Pass the complete research response JSON to this tool to save all information."
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
wiki_tool = Tool(
    name="wikipedia",
    func=wiki_tool.run,
    description="Search Wikipedia for information about a topic"
)