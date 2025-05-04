import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, AIMessage,SystemMessage
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from typing import Annotated

from fastapi import FastAPI
import uvicorn
import argparse

from dotenv import load_dotenv
_ = load_dotenv('/secrets/.env')

global_config = {}

ROUTING_PROMPT="""
Route the input to: crisis (if there is TRUE mental health emergency implicit or explicit 
(eg. suicideal ideation, self harm, etc.), rag (if there is a specific question about
GUILT), or fine_tuned (otherwise and for general conversation flow)
"""


# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["fine_tuned", "crisis", "rag"] = Field(
        None, description="The next step in the routing process"
    )

class State(TypedDict):
    input: str
    decision: str
    output: str
    messages: Annotated[list, add_messages]
    

llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-001",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

router = llm.with_structured_output(Route)

def convert_to_fine_tuned_format(messages):
    content = []
    for m in messages:
      if isinstance(m,HumanMessage):
        content.append({"from":"patient","value":m.content})
      elif isinstance(m,AIMessage):
        content.append({"from":"therapist","value":m.content})
    return content


def fine_tuned(state: State):
    import requests
    fine_tuned_url=global_config["ft_url"]+'/sft/inference'
    #fine_tuned_url=]"http://127.0.0.1:8001/fine_tuned_chat"

    data = {
            "messages": convert_to_fine_tuned_format(state["messages"])
            }
    #print(data)
    response = requests.post(fine_tuned_url, headers={"Content-Type": "application/json"}, json=data)
    #print(response.json())
    return {"messages":AIMessage(content=response.json()),"output": "fine_tuned"}


def crisis(state: State):
    print('*********************Crisis-State**********************')
    #result = llm.invoke(state["input"])
    #return {"output": result.content}
    CRISIS_MESSAGE="Dude, you need professional help! This bot ain't gonna help you!"
    return {"messages": AIMessage(content="Dude, you need professional help! This bot ain't gonna help you!"),"output": "crisis"}

def rag(state: State):
    import requests
    print('*********************RAG******************************')
    data = {"question": state["messages"][-1].content}
    #response = requests.post("http://127.0.0.1:9000/rag/str", params=data)
    response = requests.post(global_config["rag_url"]+'/rag/str', params=data)
    #print(response.text, '\n')
    return {"messages":AIMessage(content=response.text),"output": "rag"}


def llm_call_router(state: State):
    """Route the input to the appropriate node"""

    # Run the augmented LLM with structured output to serve as routing logic
    decision = router.invoke(
        [
            SystemMessage(content=ROUTING_PROMPT),
            HumanMessage(content=state["input"]),
        ]
    )

    return {"decision": decision.step}

def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "fine_tuned":
        return "fine_tuned"
    elif state["decision"] == "rag":
        return "rag"
    elif state["decision"] == "crisis":
        return "crisis"


router_builder = StateGraph(State)
router_builder.add_node("fine_tuned", fine_tuned)
router_builder.add_node("crisis", crisis)
router_builder.add_node("rag", rag)
router_builder.add_node("llm_call_router", llm_call_router)

router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
                    "llm_call_router",
                    route_decision,
                    {  # Name returned by route_decision : Name of next node to visit
                        "fine_tuned": "fine_tuned",
                        "rag": "rag",
                        "crisis": "crisis",
                    },
)
router_builder.add_edge("fine_tuned", END)
router_builder.add_edge("crisis", END)
router_builder.add_edge("rag", END)

memory = MemorySaver()
router_workflow = router_builder.compile(checkpointer=memory)
app = FastAPI()


from fastapi import FastAPI
from pydantic import BaseModel
class ChatRequest(BaseModel):
    id: str
    user_input: str

    
@app.post('/chat')
def chat(req : ChatRequest):
    config = {"configurable": {"thread_id": req.id}}
    input_messages = [HumanMessage(req.user_input)]
    
    output = router_workflow.invoke({"messages": input_messages,"input":req.user_input}, config)
    #print(f"****chat endpoint*****: {output["messages"][-1]}")
    return output["messages"][-1].content


def main():
    parser = argparse.ArgumentParser(description="Process three URL arguments.")
    parser.add_argument("--server_url", required=True, help="server URL: ip:port")
    parser.add_argument("--ft_url", required=True, help="ft URL: http://ip:port/endpoint")
    parser.add_argument("--rag_url", required=True, help="rag URL: http://ip:port/endpoint")

    args = parser.parse_args()

    global_config["server_url"] = args.server_url
    global_config["ft_url"] = args.ft_url
    global_config["rag_url"] = args.rag_url

    uvicorn.run(app, 
                host=global_config["server_url"].split(":")[0], 
                port=int(global_config["server_url"].split(":")[1]))
    
if __name__ == "__main__":
    main()