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


# llm = ChatAnthropic(model = "claude-3-5-sonnet-20241022")
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

tools = [
    search_tool, wiki_tool, save_tool
]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    # Checking if the output is already a string (direct JSON)
    if isinstance(raw_response.get("output"), str):
        json_str = raw_response.get("output")
        structured_response = json.loads(json_str)
        print(structured_response)
    # Checking if the output is in the expected format from previous versions
    elif isinstance(raw_response.get("output"), list) and len(raw_response.get("output")) > 0:
        text_content = raw_response.get("output")[0]["text"]
        structured_response = parser.parse(text_content)
        print(structured_response)
    else:
        print("Unexpected response format:", raw_response)
except Exception as e:
    # more detailed error information
    print("Error parsing response:", e)
    print("Raw Response Type:", type(raw_response.get("output")))
    print("Raw Response Content:", raw_response.get("output"))
    
    # Trying to extract directly from the response if parsing fails
    try:
        if isinstance(raw_response.get("output"), str):
            json_str = raw_response.get("output")
            parsed_json = json.loads(json_str)
            print("\nSuccessfully parsed JSON:")
            print(parsed_json)
        # If raw_response is a complex object, try to extract JSON
        elif isinstance(raw_response.get("output"), list) and len(raw_response.get("output")) > 0:
            text_content = raw_response.get("output")[0]["text"]
            # Try to extract JSON part if it exists
            if '{' in text_content and '}' in text_content:
                json_start = text_content.find('{')
                json_end = text_content.rfind('}') + 1
                json_str = text_content[json_start:json_end]
                print("\nAttempting to parse extracted JSON:")
                parsed_json = json.loads(json_str)
                print(parsed_json)
    except Exception as inner_e:
        print("Failed to extract JSON:", inner_e)