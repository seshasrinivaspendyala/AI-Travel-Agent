from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
from amadeus import Client
from langchain_community.agent_toolkits.amadeus.toolkit import AmadeusToolkit
from langchain.tools.amadeus.closest_airport import AmadeusClosestAirport
from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.tools.amadeus.flight_search import AmadeusFlightSearch
from langchain.agents import load_tools
import os
import re
import time
import streamlit as st

# Model

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=r"C:\Users\MTL2\Documents\seshu\models\GGUF_models\Meta-Llama-3.1-8B-Instruct-Q4_K_S.gguf",
    n_gpu_layers=32, 
    seed=512,
    n_ctx=8192,
    f16_kv=True,
    callback_manager=callback_manager,
    verbose=True,
    temperature=0,
    top_p=0.95,
    n_batch=512,
)
llm.client.verbose = False

#Tools

load_dotenv()

amadeus_client_secret = os.getenv("AMADEUS_CLIENT_SECRET")
amadeus_client_id = os.getenv("AMADEUS_CLIENT_ID")

search = GoogleSerperAPIWrapper()
google_search_tool = Tool(
        name="Google Search tool",
        func=search.run,
        description="useful for when you need to ask with search",
)

amadeus = Client(client_id=amadeus_client_id, client_secret=amadeus_client_secret)
amadeus_toolkit = AmadeusToolkit(client=amadeus, llm=llm)
AmadeusToolkit.model_rebuild()
AmadeusClosestAirport.model_rebuild()
AmadeusFlightSearch.model_rebuild()

tools = [google_search_tool] + amadeus_toolkit.get_tools() + load_tools(["serpapi"])

#Prompt Template

PREFIX = """[INST]Respond to the human as helpfully and accurately as possible. You have access to the following tools:"""

FORMAT_INSTRUCTIONS = """Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Use the closest_airport tool and single_flight_search tool for any flight related queries. Give all the flight details including Flight Number, Carrier, Departure time, Arrival time and Terminal details to the human.
Use the Google Search tool and knowledge base for any itinerary-related queries. Give all the detailed information on tourist attractions, must-visit places, and hotels with ratings to the human.
Use the Google Search tool for distance calculations. Give all the web results to the human.

Always consider the traveler's preferences, budget constraints, and any specific requirements mentioned in their query.

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{{{
  "action": "Final Answer",
  "action_input": "Provide the detailed Final Answer to the human"
}}}}
```[/INST]"""

SUFFIX = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
Thought:[INST]"""

HUMAN_MESSAGE_TEMPLATE = "{input}\n\n{agent_scratchpad}"

#Agent

agent = StructuredChatAgent.from_llm_and_tools(
    llm,
    tools,
    prefix=PREFIX,
    suffix=SUFFIX,
    human_message_template=HUMAN_MESSAGE_TEMPLATE,
    format_instructions=FORMAT_INSTRUCTIONS,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=1, early_stopping_method='generate')

# Main title and description
st.title(":earth_africa::airplane: AI Travel Agent")
st.write("This Langchain-powered AI Travel Agent is designed to assist you with quick travel-related queries. You can request **flight details** for a specific day or find **nearby airports** by location. For other questions, we use **Google Search** for the latest information.")
# Sidebar with questions
st.sidebar.title(":bulb: Example Queries")
st.sidebar.write("Here are some questions you can ask:")
st.sidebar.markdown("""
- **What are the major airlines that operate to London?**
- **Can you give me the best places to celebrate Holi in India?**
- **What is the distance between eiffel tower and the Paris CDG airport?**
- **What are the flight details of the cheapest flight from Hyderabad to Udaipur available on 20th October 2024?**
- **What are the best places to visit in Spain?**
""")

# Section for important notes
st.markdown("#### **:warning: Important Notes**")
st.write("""
- Include your **starting location**, **destination**, and **travel date** when requesting flight details.
- Always **verify important information**, as the agent may make mistakes.
""")

# Additional instruction in a block quote
st.markdown("> **:notebook: Quick Tip:** Check the **side-bar** for more examples to guide you!")

question = st.text_area("")
if question:
    with st.spinner("Generating answer..."):
        response = agent_executor.invoke({"input": question})

        placeholder = st.empty()
        content = response['output']

        words = re.split(r'(\s+)', content)

        accumulated_text = ""

        for word in words:
            accumulated_text += word + " "
            placeholder.write(accumulated_text)
            time.sleep(0.01)