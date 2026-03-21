import streamlit as st
import os
from dotenv import load_dotenv

# Standard 2026 Imports
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler 

# Using the unified AgentExecutor for the ReAct pattern
from langchain_classic.agents import create_react_agent, AgentExecutor  
from langchain_core.prompts import PromptTemplate

load_dotenv()

st.set_page_config(page_title="LangChain Search", page_icon="🔎")
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

duck_wrapper = DuckDuckGoSearchAPIWrapper()
search = DuckDuckGoSearchRun(api_wrapper=duck_wrapper)

tools = [search, arxiv, wiki]

# --- REACT PROMPT TEMPLATE (The Critical Fix) ---
# ReAct agents MUST have these exact variables in the string: {tools}, {tool_names}, {agent_scratchpad}
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

# --- Chat Session Management ---
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Using 'user_query' instead of 'prompt' to avoid naming conflicts
if user_query := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    # Initialize LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", streaming=True)
    
    # 1. Create the Agent Logic (The "Brain")
    # We pass the prompt_template OBJECT here, not a string.
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_template
    )

    # 2. Create the Agent Executor (The "Body")
    # 'verbose' and 'handle_parsing_errors' live here!
    search_agent = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
         
        # We invoke the agent with the user's text
        response = search_agent.invoke(
            {"input": user_query}, 
            config={"callbacks": [st_cb]}
        )
        
        final_answer = response["output"]
        
        st.session_state.messages.append({'role': 'assistant', "content": final_answer})
        st.write(final_answer)
