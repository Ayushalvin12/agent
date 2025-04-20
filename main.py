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
    # Parsing the the response
    if isinstance(raw_response.get("output"), str):
        structured_response = json.loads(raw_response.get("output"))
    elif isinstance(raw_response.get("output"), list) and raw_response.get("output"):
        text_content = raw_response.get("output")[0]["text"]
        structured_response = parser.parse(text_content)
    else:
        structured_response = raw_response.get("output")
    
    print(json.dumps(structured_response, indent=2))
    
    # tracking which tools were actually used
    actual_tools_used = []
    if "intermediate_steps" in raw_response:
        for step in raw_response["intermediate_steps"]:
            if len(step) >= 1 and hasattr(step[0], "tool"):
                tool_name = step[0].tool
                if tool_name not in actual_tools_used:
                    actual_tools_used.append(tool_name)
    
    # Update tools_used with actual tools
    if actual_tools_used and isinstance(structured_response, dict):
        structured_response["tools_used"] = actual_tools_used
    
    # saving to a file
    save_tool.func(json.dumps(structured_response, indent=2))
    print("Research data saved successfully.")
    
except Exception as e:
    print(f"Error: {e}")
    # Basic fallback - to save whatever it can
    try:
        output = raw_response.get("output")
        if isinstance(output, str) and "{" in output and "}" in output:
            json_start = output.find('{')
            json_end = output.rfind('}') + 1
            json_str = output[json_start:json_end]
            save_tool.func(json_str)
        else:
            save_tool.func(str(output))
    except:
        print("Could not save research data.")