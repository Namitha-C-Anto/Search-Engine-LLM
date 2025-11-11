import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun
#from langchain.agents import initialize_agent, agent_types
from langchain.agents import initialize_agent, AgentType

from langchain.callbacks import StreamlitCallbackHandler ## allows you to communicate with all this tools
import os
from dotenv import load_dotenv
load_dotenv()

##Arxiv Tools Creation
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

##Wikipedia Tools Creation
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

##Creating DuckDuck Search Tool
search = DuckDuckGoSearchRun(name="Search")

##Streamlit APP Creation
st.title("Langchain - Chat with Search")
"""
In this example, we are using 'StreamlitCallbackHandler' to display the thougts and actions.
Try more Langchain - Streamlit Agent examples at [github.com/langchain-ai/streamlit-agents]
"""
## Sidebar for Settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")

##create session state: give important message roles.
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "Assistant", "content":"Hi, I am a chatbot who can search the web. How can I a help you?"}
    ]

#for msg in st.session_state.messages:
#    st.chat_message(msg["role"]).write(msg["content"])


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(str(msg["content"]))

    #st.chat_input(...): This is a function from the Streamlit library that renders 
    #a text input box at the bottom of a chat interface in a web application. 
    #It waits for the user to type a message and press Enter.

if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name= "llama-3.1-8b-instant", streaming=True)

    #tools = [search, arxiv, wiki]
    tools = [search]#, arxiv, wiki]

                                                ## based on how they handle the context & memory
   # search_agent= initialize_agent(tools=tools, llm= llm, agent=agent_types.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors = True )
    search_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True
)

    from langchain.agents import create_react_agent, AgentExecutor
    from langchain import hub

    # 1. Load the default ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    # 2. Create a ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # 3. Wrap the agent in an executor (to actually run it)
    search_agent = AgentExecutor(agent=agent, tools=tools, verbose=True) 
    ## 4. Run the agent
    #response = search_agent.invoke({"input": "What is LangChain?"})
    #print(response["output"])

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        #response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        response = search_agent.invoke({"input": st.session_state.messages}, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content": response})
        #st.write(response)
        st.write(str(response))
