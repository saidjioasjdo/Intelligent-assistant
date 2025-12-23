import asyncio
import os
import time
from time import sleep
from FlagEmbedding import FlagReranker
import uvicorn
from dominate import document
from fastapi.exceptions import HTTPException
from langchain.agents import create_react_agent, AgentExecutor
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, Query, Form, File
from fastapi.responses import StreamingResponse
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from pydantic import BaseModel

from chat_model import model, history_summary, embedding_model
from Agent_tools import tools
from chat_prompt_and_memory import prompt1, add_documents_to_prompt
from db import connect_mysql

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
app = FastAPI(title="chat_with_llm")

origins = [
    "*",  # 允许所有域名访问
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许的来源列表
    allow_credentials=True,  # 是否允许携带凭据（cookies等）
    allow_methods=["*"],  # 允许的HTTP方法（GET, POST等）
    allow_headers=["*"],  # 允许的请求头
)


class User(BaseModel):
    name: str
    password: str


@app.post("/api/register")
async def register(user: User):
    try:
        connect = connect_mysql()
        cursor = connect.cursor()
        sql = f'INSERT INTO user(name,password) VALUES("{user.name}","{user.password}");'
        cursor.execute(sql)
        return {"code": 200, "detail": 1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"错误为:{e}")


@app.post("/api/login")
async def login(user: User):
    try:
        connect = connect_mysql()
        cursor = connect.cursor()
        sql = f'SELECT uid,name,password FROM user WHERE name="{user.name}" AND password="{user.password}";'
        cursor.execute(sql)
        data = cursor.fetchall()
        if data == ():
            return {"code": 200, "detail": 0}
        else:
            return {"code": 200, "detail": 1, "uid": data[0][0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"错误为:{e}")


@app.post("/api/chat/load_file")
async def get_data_from_file(question: str = Form(...), file: UploadFile = File):
    try:
        save_dir = "./Temp"
        os.makedirs(save_dir, exist_ok=True)
        file_path = save_dir + "/" + file.filename
        suffix = os.path.splitext(file_path)[1].lower()
        with open(f"Temp/{file.filename}", "wb") as f:
            f.write(await file.read())
        if suffix == ".pdf":
            loader = PyPDFLoader(file_path=file_path, mode="page")
        elif suffix == ".txt":
            loader = TextLoader(file_path=file_path, encoding='utf-8')
        elif suffix == ".csv":
            loader = CSVLoader(file_path=file_path, encoding='utf-8')
        else:
            return {"code": 500, "page_content": "暂时不支持该类文档的分析"}
        data_load = loader.load()
        spliter = RecursiveCharacterTextSplitter(
            keep_separator=True,
            chunk_size=2000,
            chunk_overlap=20,
            add_start_index=True
        )
        data = spliter.split_documents(data_load)
        db = Chroma.from_documents(documents=data, embedding=embedding_model,persist_directory='./database')
        retriever = db.as_retriever(search_type="similarity_score_threshold",
                                    search_kwargs={"score_threshold": 0.1, "k": 5})
        result = retriever.invoke(question)
        page_content = []
        for i in range(len(result)):
            page_content.append(result[i].page_content)
        reranker = FlagReranker(model_name_or_path='./model/reranker',
                                use_gpu=True, device='cuda',
                                use_fp16=True)
        scores = reranker.compute_score([(question, content) for content in page_content])
        re_indices = np.argsort(scores)[::-1]
        rerank_page_content = [page_content[i] for i in re_indices[:10]]
        result = "\n\n".join(rerank_page_content)
        os.remove(file_path)
        return {"code": 200, "page_content": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"错误为{e}")


async def stream_agent_response(question: str, agent_executor: AgentExecutor):
    config = RunnableConfig(
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    async for chunk in agent_executor.astream(input={"input": question}, config=config):
        content = chunk["output"]
        print(content, end="", flush=True)
        if content:
            sse_data = f"data:{content}\n\n"
            yield sse_data.encode("utf-8")
            await asyncio.sleep(0.1)
    yield "data: [DONE]\n\n".encode("utf-8")


@app.get("/api/chat/get")
async def chat_with_model(question: str, content: str):
    try:
        if content == "No content":
            prompt = prompt1
        else:
            prompt = add_documents_to_prompt(content)
        agent = create_react_agent(
            llm=model,
            prompt=prompt,
            tools=tools,

        )
        agent_executor = AgentExecutor(
            agent=agent,
            memory=memory,
            verbose=False,
            tools=tools,
            handle_parsing_errors=True,

        )
        response = agent_executor.invoke(input={"input": question})
        output = response["output"]
        return_data = {
            "code": 200,
            "detail": "成功响应",
            "message": output,
        }
        stream_Data = stream_agent_response(question, agent_executor)
        # return StreamingResponse(content=stream_Data, media_type="text/event-stream")
        return return_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"错误为{e}")


class History(BaseModel):
    chat_id: int
    human_message: list[str]
    ai_message: list[str]
    uid: int
    whether_load_history: int


@app.post("/api/chat/store_history")
async def submit_history_to_model(history: History):
    connect = connect_mysql()
    cur = connect.cursor()
    try:
        if history.chat_id == 0:
            sql_search_chat_id = f"select MAX(chat_id) from chat_history;"
            cur.execute(sql_search_chat_id)
            data = cur.fetchall()
            if data[0][0] is None:
                chat_id = 1
            else:
                chat_id = data[0][0] + 1
        else:
            chat_id = history.chat_id
        if history.whether_load_history == 1:
            print('处于历史状态，删除所选的')
            sql_del_hs = f"DELETE FROM history_summary WHERE chat_id={chat_id} AND uid={history.uid};"
            sql_del_ch = f"DELETE FROM chat_history WHERE chat_id={chat_id} AND uid={history.uid};"
            cur.execute(sql_del_ch)
            cur.execute(sql_del_hs)

        for index in range(len(history.human_message)):
            sql = f'INSERT INTO chat_history (chat_id,AI_message,Human_message,uid) VALUES ({chat_id},"{history.ai_message[index]}","{history.human_message[index]}",{history.uid});'
            cur.execute(sql)

        summary = history_summary(history.human_message[0], history.ai_message[0])

        sql1 = f'INSERT INTO history_summary (summary,uid,chat_id) VALUES ("{summary}",{history.uid},{chat_id});'
        cur.execute(sql1)

        return {"success": True, "summary": summary, "message": "成功写入数据库", "chat_id": chat_id}
    except Exception as e:
        connect.rollback()
        raise HTTPException(status_code=500, detail=f"服务器错误：{str(e)}")
    finally:
        connect.close()


@app.get("/api/chat/get_summary")
async def load_summary(uid: int):
    connect1 = connect_mysql()
    cur = connect1.cursor()
    try:
        sql = f'SELECT chat_id,summary FROM history_summary WHERE uid={uid};'
        cur.execute(sql)
        summary_history = cur.fetchall()
        response = []
        for data in summary_history:
            body = {"chat_id": data[0], "summary": data[1]}
            response.append(body)
        return {"code": 200, "detail": "加载历史成功", "message": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史记录失败：{str(e)}")
    finally:
        connect1.close()


@app.get("/api/chat/get_history_by_id")
async def get_history_by_id(chat_id: int, uid: int):
    connect2 = connect_mysql()
    cur = connect2.cursor()
    try:
        sql = f'SELECT Human_message,AI_message FROM chat_history WHERE chat_id={chat_id} AND uid={uid};'
        cur.execute(sql)
        history = cur.fetchall()
        human_history = []
        ai_history = []
        memory.chat_memory.clear()
        for h in history:
            human_history.append(h[0])
            ai_history.append(h[1])
            memory.save_context(inputs={"human": h[0]}, outputs={"ai": h[1]})
        return {"code": 200, "detail": "加载所选对话历史成功", "human_history": human_history, "ai_history": ai_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取当前对话历史记录失败：{str(e)}")
    finally:
        connect2.close()


if __name__ == '__main__':
    uvicorn.run(
        app=app,
        host="127.0.0.1",
        port=8000
    )
