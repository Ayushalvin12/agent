import json
import textwrap
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_txt(data: str, filename: str = "research_output.txt"):
    # Parse the data if it's a JSON string
    if isinstance(data, str):
        try:
            if data.strip().startswith("{") and data.strip().endswith("}"):
                parsed_data = json.loads(data)
            else:
                parsed_data = data
        except:
            parsed_data = data
    else:
        parsed_data = data
    
    # Format the output with text wrapping
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_lines = [f"--- Research Output ---", f"Timestamp: {timestamp}", ""]
    
    # Handle dictionary/JSON data
    if isinstance(parsed_data, dict):
        for key, value in parsed_data.items():
            output_lines.append(f"{key}:")
            
            # Handle list values
            if isinstance(value, list):
                for item in value:
                    wrapped_item = textwrap.fill(str(item), width=80)
                    # Indent the wrapped text
                    for line in wrapped_item.split('\n'):
                        output_lines.append(f"  - {line}")
            
            # Handle string values with wrapping
            elif isinstance(value, str):
                wrapped_text = textwrap.fill(value, width=80)
                # Indent the wrapped text
                for line in wrapped_text.split('\n'):
                    output_lines.append(f"  {line}")
            
            # Handle other types
            else:
                output_lines.append(f"  {value}")
            
            # Add a blank line between fields
            output_lines.append("")
    
    # Handle string data that's not JSON
    else:
        wrapped_text = textwrap.fill(str(parsed_data), width=80)
        output_lines.extend(wrapped_text.split('\n'))
    
    # Join all lines and add final newline
    formatted_text = '\n'.join(output_lines) + "\n\n"
    
    # Write to file
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

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
wiki_tool = Tool(
    name="wikipedia",
    func=wiki_tool.run,
    description="Search Wikipedia for information about a topic"
)