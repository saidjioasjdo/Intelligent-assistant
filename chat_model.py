import dotenv
import os

from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv(dotenv_path=".env")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model_name="Qwen/Qwen2.5-7B-Instruct",)
embedding_model=OpenAIEmbeddings(model="Qwen/Qwen3-Embedding-0.6B")


def history_summary(human_message, ai_message: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ('system', "你是一名精通内容主旨概括的专家。"),
        ('human', """现在给你一段对话，这是用户的提问：“{human_message}”，这是大模型回复的内容：“{ai_message}”。
    请你像**给一篇文章拟定标题**一样，用10个字左右的短语，**概括出这段对话的中心思想和主题**。""")
    ])
    chain = LLMChain(llm=model, verbose=False, prompt=prompt)
    response = chain.invoke(input={"human_message": human_message, "ai_message": ai_message})
    return response['text']
# 测试功能
# if __name__ == '__main__':
#     response=history_summary("给我讲解一下什么是langchain","对于全栈初学者（尤其是计算机相关专业学生），LangChain 可以理解为 “大语言模型（LLM）的开发框架” —— 就像用 Django/Flask 开发 Web 应用、用 PyTorch/TensorFlow 开发深度学习模型一样，LangChain 是专门用来搭建基于 LLM 的应用（如聊天机器人、问答系统、数据分析工具等）的 “工具箱”，让你不用从零处理 LLM 的调用、数据交互、流程串联等底层工作。")
#     print(response)
