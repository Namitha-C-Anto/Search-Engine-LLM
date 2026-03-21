import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
 
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks import StreamlitCallbackHandler 
 
from langchain.agents import create_agent 
from langchain_core.prompts import ChatPromptTemplate
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

duck_wrapper = DuckDuckGoSearchAPIWrapper()
search = DuckDuckGoSearchRun(api_wrapper=duck_wrapper)

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
    
    # Use the 2026 standard model
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", streaming=True)
    
    # --- Modern Agent Logic --- 
    search_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful assistant. Use tools to verify facts before answering.",
        debug=True # This replaces 'verbose=True'
    )

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        
        # We invoke the agent directly (no AgentExecutor wrapper needed)
        # Note: 2026 standard uses 'messages' key for the conversation
        response = search_agent.invoke(
            {"input": prompt}, 
            config={"callbacks": [st_cb]}
        )
        
        # Extract response from the modern message-based output
        final_answer = response["output"]
        
        st.session_state.messages.append({'role': 'assistant', "content": final_answer})
        st.write(final_answer)
