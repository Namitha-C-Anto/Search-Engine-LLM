import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables (like API keys if they're in .env file)
load_dotenv()

# --- New Imports for Modular LangChain ---
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler 

# Core Agent Imports (new style)
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
# ----------------------------------------


st.title("🔎 LangChain - Chat with Search")
"""
This application uses the modern LangChain Expression Language (LCEL) and the `create_react_agent` approach for resilient deployment.
"""

## Sidebar for settings
st.sidebar.title("Settings")
# Use st.secrets or load_dotenv for API key, but handling it via sidebar is retained for local testing
# Get API key from sidebar or environment variable
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password", 
                                     value=os.getenv("GROQ_API_KEY", ""))

# Ensure API key is available before proceeding
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
# ------------------


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])


if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Initialize LLM and Agent inside the chat block
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", streaming=True)
    
    
    # --- New Agent Logic (Replaces initialize_agent) ---
    # 1. Define the Prompt Template (Standard ReAct format)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Use the provided tools to answer questions."),
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # 2. Create the Agent
    agent = create_react_agent(llm, tools, prompt_template)

    # 3. Create the Executor
    search_agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # ----------------------------------------------------


    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        
        # The .invoke method is the standard way to call agents and chains now
        response = search_agent.invoke({"input": prompt}, config={"callbacks": [st_cb]})
        
        # The result is a dictionary, extract the final answer
        final_answer = response.get("output", "Could not retrieve an answer.")
        
        st.session_state.messages.append({'role': 'assistant', "content": final_answer})
        st.write(final_answer)
