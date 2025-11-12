# git clone "https://github.com/anweshabose2025/RAG_QA_Chatbot_GenAI_Project.git"
# streamlit run 1-Streamlit_app.py
# Python == 3.10

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper # type: ignore
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun # type: ignore
from langchain.agents import create_openai_tools_agent, AgentExecutor # type: ignore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings # type: ignore
from langchain.tools.retriever import create_retriever_tool # type: ignore
from langchain.vectorstores.faiss import FAISS # type: ignore
from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_core.chat_history import BaseChatMessageHistory # type: ignore
from langchain_core.runnables.history import RunnableWithMessageHistory # type: ignore
from langchain_community.chat_message_histories import ChatMessageHistory # type: ignore
from langchain_core.output_parsers import StrOutputParser

# Streamlit setup
st.set_page_config(page_title="LangChain Search Chat", page_icon="🔎")
st.title("🔎 LangChain - Chat with Search")

# Sidebar for API keys
st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
model=st.sidebar.selectbox("Select Open Source model",["openai/gpt-oss-120b","openai/gpt-oss-20b"])
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7) # value=0.7: default temperature
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150) # value=150: default max_tokens

if not groq_api_key:
    st.warning("⚠️ Please enter the Groq API keys to continue.")
    st.stop()

user_input = st.chat_input("Ask anything...")

# Create Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper,name="arxiv")
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper,name="wikipedia")
search_tool = DuckDuckGoSearchRun(name="search")
tools = [search_tool, arxiv_tool, wiki_tool]

# Load PDF and create retriever tool
uploaded_files = st.file_uploader("Choose PDF file(s) to enable document search", type="pdf", accept_multiple_files=True)
if st.button("Learn Document") and uploaded_files and "retriever" not in st.session_state:
    with st.spinner("Learning Documents. Please Wait."):
        documents = []
        for file in uploaded_files:
            with open(f"_{file.name}", "wb") as f:
                f.write(file.read())
            doc = PyPDFLoader(f"_{file.name}").load()
            documents.extend(doc)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.from_documents(splits,embeddings)
        st.session_state.retriever = vectorstore.as_retriever()
        st.success("Learned Document successfully. Now you can ask me anything from the uploaded document(s).")

if "retriever" in st.session_state:
    retriever_tool = create_retriever_tool(st.session_state.retriever, name="pdf-search", description="Search from uploaded PDFs")
    tools.append(retriever_tool)
    
# Initialize LLM
llm = ChatGroq(api_key=groq_api_key, model_name=model)

# Prompts
prompt = ChatPromptTemplate.from_messages([
    ("system", "If pdf-search tool is present, search from the pdf-search tool only. "
    "If needed formulate the question and search from pdf-search tool only. Or else do search from "
    "website, wikipedia, archive search tool. If answer is not in website, wikipedia or archive, "
    "answer must be in the pdf-search tool only. Please give an appropriate, "
    "understandable response. If answer is not known, say that you dont know the answer."),
    MessagesPlaceholder("chat_history"), # for chat history
    MessagesPlaceholder("agent_scratchpad"), # for agent
    ("user", "{input}")
])

# Output Parser
output_parser = StrOutputParser()

# Agent
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Session-aware history
if "store" not in st.session_state:
    st.session_state.store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Wrap RAG with history
runnable_msg_history = RunnableWithMessageHistory(agent_executor,get_session_history,input_messages_key="input", history_messages_key="chat_history")

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    response = runnable_msg_history.invoke({"input": user_input}, config={"configurable": {"session_id": "chat_1"}})
    final_answer = response["output"]
    #final_answer = response.get("output", "Sorry, I couldn't find an answer.")
    st.session_state.messages.append({"role": "assistant", "content": final_answer})
    st.chat_message("assistant").write(final_answer)
    st.write("Thank You. I hope it helped. Dont hesitate to ask the next question. 😊")
    st.warning("[[ If you want me to learn any Other Document, please refresh the page and upload the Document entering the API Key. Otherwise, no need to upload the same file again ]]")