import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# --- 2026 Standard Imports ---
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler 
 
from langchain.agents import create_agent 
from langchain_core.prompts import PromptTemplate         # Standard for ReAct
# -----------------------------

st.title("🔎 LangChain - Chat with Search")

## Sidebar Settings
st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password", 
                                     value=os.getenv("GROQ_API_KEY", ""))

if not groq_api_key:
    st.error("Please enter your Groq API Key in the sidebar.")
    st.stop()

# --- Tool Setup ---
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")
tools = [search, arxiv, wiki]

# --- Chat Session Management ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Initialize LLM with streaming enabled
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", streaming=True)
    
    # --- ReAct Agent Logic ---
    # ReAct agents require a very specific prompt structure with these exact variables
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt_template = PromptTemplate.from_template(template)

    # Create the Agent and Executor
    agent = create_agent(llm, tools, prompt_template)
    search_agent = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        # The CallbackHandler creates the "thinking" expanders in Streamlit
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        
        # Invoke the agent
        response = search_agent.invoke(
            {"input": prompt}, 
            config={"callbacks": [st_cb]}
        )
        
        final_answer = response.get("output", "I encountered an error while searching.")
        st.session_state.messages.append({'role': 'assistant', "content": final_answer})
        st.write(final_answer)
