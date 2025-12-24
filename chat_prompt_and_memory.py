from langchain.chains.qa_with_sources.stuff_prompt import template
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain import hub

prompt2=PromptTemplate.from_template(
    template="""
    Answer the following questions as best you can. You have access to the following tools:
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
Thought:{agent_scratchpad}
    """
)
prompt1=hub.pull("hwchase17/react-chat")


def add_documents_to_prompt(document:str):
    prompt_withRAG = PromptTemplate.from_template(
        """
  “Answer the following questions as best you can. You have access to the following tools:
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
Content: {content}（包含从文件中读取的数据，模型必须依据这些数据回答问题）
Thought:{agent_scratchpad}”
        """
    ).partial(content=document)
    return prompt_withRAG
# if __name__ == '__main__':
#     p=prompt_withRAG.format(tools="a",input="b",agent_scratchpad="d")
#     print(p)
# memory=ConversationSummaryBufferMemory(
#     llm=model,
#     max_token_limit=500,
#     return_messages=True,
#     memory_key="chat_history"
# )