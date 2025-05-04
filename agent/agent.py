import requests
import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import argparse
from fastapi.responses import PlainTextResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or you can restrict this to 'http://localhost:3000')
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class QueryInput(BaseModel):
    question: str

# @app.post("/agent/rag")
# def receive_from_rag(input: QueryInput):
#     data = {"question": input.question}
#     response = requests.post("http://rag:9000/rag/str", params=data)
#     return PlainTextResponse(response.text)


# class SFTInput(BaseModel):
#     content: str

# @app.post("/agent/sft")
# def receive_from_sft(input: SFTInput):
#     data = {
#         "messages": [
#             {
#                 "from": "patient",
#                 "value": input.content
#             }
#                     ]
#             }
#     response = requests.post("http://sft:9000/sft/inference", headers={"Content-Type": "application/json"}, json=data)
#     return response.json()['reply']


################### Sandra's backend #####################
class MessagePayload(BaseModel):
    message: str
    sessionId: str

@app.post("/api/message")
async def receive_message(payload: MessagePayload):
    if payload.message[:4].lower() == "@rag":
        # print(payload.message[5:].lower() )
        data = {"question": payload.message[4:].strip()}
        response = requests.post("http://rag:9000/rag/str", params=data)
        # print(response.text)
        return {"text": response.text}
    
    elif payload.message[:4].lower() in ["@sft"]:
        # print(payload.message[5:].lower())
        data = {
            "messages": [
                {
                    "from": "patient",
                    "value": payload.message[4:].strip()
                }
                        ]
                }
        reply = requests.post("http://sft:9000/sft/inference", headers={"Content-Type": "application/json"}, json=data)
        # print(reply.json()['reply'])
        return { "text": reply.json()['reply'] }
    else:
        pass

##########################################################

def main():
    while True:
        method = input("rag or sft: ")
        if method == "rag":
            query = input("Query: ")
            if query.lower() in ["quit", "exit", "q"]:
                print("No query provided")
                pass
            else:
                data = {"question": query}
                response = requests.post("http://rag:9000/rag/str", params=data)
                print(response.text, '\n')

        elif method == "sft":
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            data = {
                "messages": [
                    {
                        "role": "string",
                        "content": user_input
                    }
                ]
            }
            response = requests.post("http://sft:9000/sft/inference", headers={"Content-Type": "application/json"}, json=data)
            print(response.json()['reply'], '\n')
        elif method == "exit":
            print("Goodbye!")
            break
            
        else:
            print("Invalid method. Please enter 'rag' or 'sft'.")

       
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI")
    parser.add_argument(
        "--test_agent",
        action="store_true",
        help="Test Agent for RAG and SFT",
    )
    args = parser.parse_args()
    if args.chat:
        main()
    