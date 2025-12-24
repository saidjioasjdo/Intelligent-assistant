import os
import dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

dotenv.load_dotenv(dotenv_path=".env")
os.environ["OPENAI_BASE_URL"]=os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
#RAG_tools
loader=TextLoader(file_path="./goods.txt",encoding="utf-8")
data=loader.load()
spliter=RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=1000,
    keep_separator=True,
)
data=spliter.split_documents(documents=data)
embedding_model=OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-0.6B")
db=Chroma.from_documents(
    documents=data,
    embedding=embedding_model,
    persist_directory="./database"
)
retriever=db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold":0.1}
)
rag_tool=create_retriever_tool(
    retriever=retriever,
    name="knowledge_Search",
    description="当用户需要查询产品A或者B的相关信息，以及公司政策的时候使用此工具"
)
#Search_tools
from langchain.tools import Tool
from langchain_community.tools import TavilySearchResults

os.environ['tavily_api_key']=os.getenv('tavily_api_key')
ts=TavilySearchResults(max_results=3)
search_tool=Tool(
    func=ts,
    name="search",
    description="A search engine optimized for real-time and current information. Use this tool to get the current date, latest news, weather, or any other information that requires up-to-date knowledge."
)
tools=[rag_tool,search_tool]