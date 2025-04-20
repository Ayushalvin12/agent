import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_ollama import ChatOllama
 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOllama(model = "llama3.2")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools.
            
            YOU MUST FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS WITH NO NESTED OBJECTS:
            {{
                "topic": "The topic being researched",
                "summary": "A detailed summary of findings",
                "sources": ["source1", "source2", "etc"],
                "tools_used": ["tool names used"]
            }}
            
            Do not include any nested JSON objects or arrays beyond what's specified.
            The 'sources' field should be a simple list of strings with your source URLs or references.
            The 'tools_used' field should be a simple list of string names of the tools you used.
            
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    # Handle string output (direct JSON)
    if isinstance(raw_response.get("output"), str):
        json_str = raw_response.get("output")
        structured_response = json.loads(json_str)
    # Handle list output format
    elif isinstance(raw_response.get("output"), list) and raw_response.get("output"):
        text_content = raw_response.get("output")[0]["text"]
        structured_response = parser.parse(text_content)
    # Fallback to raw output
    else:
        structured_response = raw_response.get("output")
        
    # Print the structured response
    print(json.dumps(structured_response, indent=2))
    
    # Save the research data to file
    save_data = json.dumps(structured_response, indent=2)
    save_tool.func(save_data)
    print("Research data saved to file successfully.")
        
except Exception as e:
    print(f"Error processing response: {e}")
    # Save raw response as fallback
    save_tool.func(str(raw_response))
    print("Saved raw response to file as fallback.")