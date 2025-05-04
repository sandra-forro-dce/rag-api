import os
import traceback
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
import argparse
import json
import shutil
from unsloth import FastLanguageModel #, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
import torch
from langchain.memory import ConversationBufferMemory
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning  # Import the specific warning class
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
import gc
# from transformers import TextStreamer

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, ToolConfig

from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# cred_path = os.path.join("..", "secrets", "couchgpt-rag-service-account.json")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

# cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "/app/secrets/couchgpt-rag-service-account.json")
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

GCP_PROJECT = "forward-map-449500-e6"
GCP_LOCATION = "us-central1"
GENERATIVE_MODEL = "gemini-1.5-flash-002"

vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are a compassionate and professional therapist helping patients with emotional and mental well-being. Limit your responses to one to two sentences maximum unless you are asked by user to give explanation. You can also give details on healing methods. Always greet the patient and say "Welcome to the Couch". When the user wishes to end the chat, tell him something encouraging as last words related to the problem they discussed and at the end say "... Now get out of the Couch! :) " . 
"""
generative_model = GenerativeModel(
	GENERATIVE_MODEL,
	system_instruction=[SYSTEM_INSTRUCTION]
)

# def inp_method():
#     input_key = 0
#     while input_key not in ['1', '2', '3', '4', '5']:
#         input_key = input("Enter epoch [1-5]: ").strip()
#     return input_key

def download(epoch_name):
    print(f"Downloading model {epoch_name} into cache\n")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name = epoch_name, 
                                                         max_seq_length = 2048, 
                                                         dtype = None, 
                                                         load_in_4bit = True)
    print("\n\n Download complete")
    return model, tokenizer

def load(epoch_name, epochs_folder, save_dir="./sft"):
    final_path = os.path.join(save_dir, epochs_folder)
    if os.path.exists(final_path):
        files = os.listdir(final_path)
        if files:
            print(f"Directory '{final_path}' contains {len(files)} file(s):")
            for f in files:
                print(f" - {f}")
            print(f"Loading existing model from '{final_path}'...")
            loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(
                model_name = f"{final_path}",
                max_seq_length = 2048,
                dtype = None,
                load_in_4bit = True,)
            return loaded_model, loaded_tokenizer
            
        else:
            model, tokenizer = download(epoch_name)
            model.save_pretrained(final_path, safe_serialization=True)
            tokenizer.save_pretrained(final_path)
            return model, tokenizer
    else:    
        os.makedirs(final_path, exist_ok=True)
        model, tokenizer = download(epoch_name)
        model.save_pretrained(final_path, safe_serialization=True)
        tokenizer.save_pretrained(final_path)
        return model, tokenizer
    
def clear_screen():
    # Check for Windows
    if os.name == 'nt':
        os.system('cls')
    # MacOS or Linux
    else:
        os.system('clear')

def cleanup_model():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def generate_reply(user_input, memory, model, tokenizer, max_output_tokens=512):
    try:
        history = memory.load_memory_variables({})["history"]
        messages = []
        if history:
            lines = history.strip().split("\n")
            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    messages.append({"from": "patient", "value": line})
                else:
                    messages.append({"from": "therapist", "value": line})
        messages.append({"from": "patient", "value": user_input})
        model = model.to("cuda")
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenizer=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        inference_model = FastLanguageModel.for_inference(model).to("cuda")
        outputs = inference_model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=max_output_tokens,
        # pad_token_id=tokenizer.eos_token_id,   # Optional but recommended
        # eos_token_id=tokenizer.eos_token_id,   # Optional but recommended
        use_cache=True
        )
        decoded_output = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        memory.save_context({"input": user_input}, {"output": decoded_output})
        return decoded_output

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return "Sorry, the model ran out of memory. Please try again with a shorter message or restart the application."
        else:
            print(f"Error in generate_reply: {str(e)}")
            print(traceback.format_exc())
            return "An error occurred while generating a reply. Please try again."
        
# def check_gpu():
#     nvidia_smi_output = os.popen("nvidia-smi").read()
#     if "No running processes found" in nvidia_smi_output:
#         print("No GPU processes found.")

#     else:
#         print("GPU processes found.\n")
#         print(nvidia_smi_output)


#### FastAPI route for inference
app = FastAPI()
model = None
tokenizer = None

class Message(BaseModel):
    from_: str = Field(..., alias="from")
    value: str

class ChatRequest(BaseModel):
    messages: list[Message] 

# @app.on_event("startup")
@asynccontextmanager
async def load_model_on_startup(app: FastAPI):
    global model, tokenizer
    models = {
        '1': 'test733/llama_3_1_8B_Instruct_lorav1_epochs1_batch8_r16_a16',
        '2': 'test733/llama_3_1_8B_epochs2_batch8_r16_a16',
        '3': 'test733/llama_3_1_8B_Instruct_epochs3_batch8_r16_a16',
        '4': 'test733/llama_3_1_8B_Instruct_epochs4_batch8_r16_a16',
        '5': 'test733/llama_3_1_8B_Instruct_epochs5_batch8_r16_a16'
    }

    epoch_name = models['5']
    epochs_folder = "epochs5"
    print('Loading model, please wait...')
    model, tokenizer = load(epoch_name, epochs_folder)
    tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
    mapping = {"role" : "from", "content" : "value", "user" : "patient", "assistant" : "therapist"},
    )
    print('Model is ready for inference!')
    yield

app = FastAPI(lifespan=load_model_on_startup)

@app.post("/sft/inference")
async def generate_inference(req: ChatRequest):
    global model, tokenizer
    max_output_tokens=512
    # messages = [m.model_dump() for m in req.messages]
    messages = [{"role": m.from_, "content": m.value} for m in req.messages] 
    # print(messages)
    try:
        model = model.to("cuda")
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenizer=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        inference_model = FastLanguageModel.for_inference(model).to("cuda")
        
        outputs = inference_model.generate(
        input_ids=inputs,
        max_new_tokens=max_output_tokens,
        use_cache=True
        )
        decoded_output = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        decoded_output = decoded_output.replace("AI: ", "").replace("Human: ", "").replace(".", ". ")
        return {"reply": decoded_output}

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            return {"error", "Sorry, the model ran out of memory. Please try again with a shorter message or restart the application."}
        else:
            print(f"Error in generate_reply: {str(e)}")
            print(traceback.format_exc())
            return {"error" : "An error occurred while generating a reply. Please try again."}

# === Startup Event: Load the model before handling any request ===


def main():
    models = {
        '1': 'test733/llama_3_1_8B_Instruct_lorav1_epochs1_batch8_r16_a16',
        '2': 'test733/llama_3_1_8B_epochs2_batch8_r16_a16',
        '3': 'test733/llama_3_1_8B_Instruct_epochs3_batch8_r16_a16',
        '4': 'test733/llama_3_1_8B_Instruct_epochs4_batch8_r16_a16',
        '5': 'test733/llama_3_1_8B_Instruct_epochs5_batch8_r16_a16'
    }

    epoch_name = models['5']
    epochs_folder = "epochs5"
    print('Loading model, please wait...')
    model, tokenizer = load(epoch_name, epochs_folder)
    tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
    mapping = {"role" : "from", "content" : "value", "user" : "patient", "assistant" : "therapist"},
    )
    print('Model is ready for inference')
    clear_screen()
    memory = ConversationBufferMemory(memory_key="history", input_key="input")
    patient_prompt = input("Patient: ")
    while patient_prompt.strip().lower() != "exit":
        if patient_prompt != "":
            therapist_reply = generate_reply(patient_prompt, memory, model, tokenizer)
            therapist_reply = therapist_reply.replace("AI: ", "").replace("Human: ", "").replace(".", ". ")
            print("\nTherapist: ", therapist_reply)
        patient_prompt = input("\nPatient: ")
    del model
    del tokenizer
    cleanup_model()
    memory.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="CLI Chat mode.",
    )
    args = parser.parse_args()
    if args.chat:
        main()
    
