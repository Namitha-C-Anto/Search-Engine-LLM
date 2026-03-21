# 🔎 LangChain - Autonomous Search Agent
A high-speed, reasoning-based AI search engine that uses the ReAct (Reason + Act) framework to browse the web, academic papers, and Wikipedia. Unlike standard chatbots, this agent can independently decide which external tools to use to verify facts before answering.

##🚀 Business Value
Fact Verification: Eliminates LLM hallucinations by cross-referencing real-time data from DuckDuckGo and Wikipedia.

Academic Insights: Integrated with ArXiv to pull from the latest peer-reviewed research.

Cost Efficiency: Utilizes Groq (Llama 3.3) for ultra-low latency inference, making it 10x faster than traditional cloud-based RAG systems.

##🧠 Core Architecture: The ReAct Loop
This project implements the Reason + Act pattern. Instead of a single "thought," the agent operates in a continuous loop:

Thought: The agent analyzes the user's prompt and identifies missing information.

Action: It selects the most appropriate tool (Search, Wiki, or ArXiv).

Observation: It parses the tool output and evaluates if the information is sufficient.

Final Answer: Once the criteria are met, it synthesizes a grounded response for the user.

##🛠️ Tech Stack
Orchestration: LangChain (LCEL)

LLM Engine: Groq (Llama-3.3-70b-versatile)

Web Interface: Streamlit

Information Retrieval: DuckDuckGo Search, Wikipedia API, ArXiv API

##⚙️ Installation & Setup
Clone the Repository:

Bash
git clone https://github.com/YOUR_USERNAME/search-engine-llm.git
cd search-engine-llm
Install Dependencies:

Bash
pip install -r requirements.txt
Configure Environment:
Create a .env file in the root directory:

Plaintext
GROQ_API_KEY=your_api_key_here
Run the App:

Bash
streamlit run app.py
##🔒 Security & Deployment
Credential Safety: Built-in support for st.secrets for secure production deployment on Streamlit Cloud.

Error Handling: Implements handle_parsing_errors=True to prevent agent loops during complex tool interactions.

Modular Design: The toolset can be easily expanded to include SQL databases, Jira, or private CSV files.

##📈 Future Enhancements
Stateful Memory: Implement st.session_state to allow for follow-up questions within the search context.

Multi-Agent Handoff: Transition to LangGraph to allow a "Research Agent" to hand off findings to a "Summary Agent."
