import os
import argparse
import pandas as pd
import json
import time
import glob
import hashlib
import chromadb
import shutil
from google.cloud import storage
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Literal


# Vertex AI
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, ToolConfig
from google.api_core.exceptions import InternalServerError, ServiceUnavailable, ResourceExhausted

# Langchain
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
GENERATIVE_MODEL = "gemini-1.5-flash-002"
INPUT_FOLDER = "books" 
OUTPUT_FOLDER = "outputs"
CHROMADB_HOST = "couchgpt-rag-chromadb"
CHROMADB_PORT = 8000
GCP_BUCKET = "couchgpt-bucket"


vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#python
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 8192,  # Maximum number of tokens for output
    "temperature": 0.25,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
}
# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in Psychology, mental health and self help. Your responses are based solely on the information provided in the text chunks given to you but don't mention this in your responses. Do not use any external knowledge or make assumptions beyond what is explicitly stated in these chunks.

When answering a query:
1. Carefully read all the text chunks provided.
2. Identify the most relevant information from these chunks to address the user's question.
3. Formulate your response using only the information found in the given chunks.
4. If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
5. Always maintain a professional and knowledgeable tone, befitting a psychology expert.
6. If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.

Remember:
- You are an expert in psychology and mental health, but your knowledge is limited to the information in the provided chunks.
- Do not invent information or draw from knowledge outside of the given text chunks.
- If asked about topics unrelated to mental health, politely redirect the conversation back to psychology-related subjects.
- Be concise in your responses while ensuring you cover all relevant information from the chunks.
- Be empathetic and kind like a psychologist is conversing with patient and don't accuse user of doing anything wrong. 

Your goal is to provide accurate, helpful information about cheese based solely on the content of the text chunks you receive with each query but don't tell the user that your responses are coming from chunks.
"""
generative_model = GenerativeModel(
	GENERATIVE_MODEL,
	system_instruction=[SYSTEM_INSTRUCTION]
)

book_mappings = {
	"Feeling Good The New Mood Therapy": {"author":"Dr. David Burns", "year": 2000},
	"Practical Psychology": {"author":"Karl S. Bernhardt", "year": 1945}
}

def generate_query_embedding(query):
	query_embedding_inputs = [TextEmbeddingInput(task_type='RETRIEVAL_DOCUMENT', text=query)]
	kwargs = dict(output_dimensionality=EMBEDDING_DIMENSION) if EMBEDDING_DIMENSION else {}
	embeddings = embedding_model.get_embeddings(query_embedding_inputs, **kwargs)
	return embeddings[0].values


def generate_text_embeddings(chunks, dimensionality: int = 256, batch_size=250, max_retries=5, retry_delay=5):
	# Max batch size is 250 for Vertex AI
	all_embeddings = []
	for i in range(0, len(chunks), batch_size):
		batch = chunks[i:i+batch_size]
		inputs = [TextEmbeddingInput(text, "RETRIEVAL_DOCUMENT") for text in batch]
		kwargs = dict(output_dimensionality=dimensionality) if dimensionality else {}

		# Retry logic with exponential backoff
		retry_count = 0
		while retry_count <= max_retries:
			try:
				embeddings = embedding_model.get_embeddings(inputs, **kwargs)
				all_embeddings.extend([embedding.values for embedding in embeddings])
				break
			except (InternalServerError, ServiceUnavailable, ResourceExhausted) as e:
				retry_count += 1
				if retry_count > max_retries:
					print(f"Failed to generate embeddings after {max_retries} attempts. Last error: {str(e)}")
					raise

				# Calculate delay
				wait_time = retry_delay * (2 ** (retry_count - 1))
				print(f"API error: {str(e)}. Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})...")
				time.sleep(wait_time)
		
	return all_embeddings


def load_text_embeddings(df, collection, batch_size=500):

	# Generate ids
	df["id"] = df.index.astype(str)
	hashed_books = df["book"].apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:16])
	df["id"] = hashed_books + "-" + df["id"]

	metadata = {
		"book": df["book"].tolist()[0]
	}
	if metadata["book"] in book_mappings:
		book_mapping = book_mappings[metadata["book"]]
		metadata["author"] = book_mapping["author"]
		metadata["year"] = book_mapping["year"]
   
	# Process data in batches
	total_inserted = 0
	for i in range(0, df.shape[0], batch_size):
		# Create a copy of the batch and reset the index
		batch = df.iloc[i:i+batch_size].copy().reset_index(drop=True)

		ids = batch["id"].tolist()
		documents = batch["chunk"].tolist() 
		metadatas = [metadata for item in batch["book"].tolist()]
		embeddings = batch["embedding"].tolist()

		collection.add(
			ids=ids,
			documents=documents,
			metadatas=metadatas,
			embeddings=embeddings
		)
		total_inserted += len(batch)
		print(f"Inserted {total_inserted} items...")

	print(f"Finished inserting {total_inserted} items into collection '{collection.name}'")


def download(max_files=None):
    print("📥 Starting download")

    shutil.rmtree(os.path.join(INPUT_FOLDER, "books"), ignore_errors=True)
    os.makedirs(os.path.join(INPUT_FOLDER, "books"), exist_ok=True)

    client = storage.Client()
    bucket = client.get_bucket(GCP_BUCKET)

    blobs = list(bucket.list_blobs(prefix="books/"))
    count = 0

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        print(f"Downloading: {blob.name}")
        blob.download_to_filename(blob.name)
        count += 1
        if max_files is not None and count >= max_files:
            break



def chunk(method="char-split", max_files=None):
	print("chunk()")

	os.makedirs(OUTPUT_FOLDER, exist_ok=True)

	text_files = glob.glob(os.path.join(INPUT_FOLDER, "*.txt"))
	if max_files:
		text_files = text_files[:max_files]  
	print("Number of files to process:", len(text_files))

	for text_file in text_files:
		print("Processing file:", text_file)
		filename = os.path.basename(text_file)
		book_name = filename.split(".")[0]

		with open(text_file) as f:
			input_text = f.read()

		text_chunks = None
		if method == "char-split":
			chunk_size = 350
			chunk_overlap = 20
			text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator='', strip_whitespace=False)
			text_chunks = text_splitter.create_documents([input_text])
		elif method == "recursive-split":
			chunk_size = 350
			text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
			text_chunks = text_splitter.create_documents([input_text])

		text_chunks = [doc.page_content for doc in text_chunks]
		print("Number of chunks:", len(text_chunks))

		if text_chunks:
			data_df = pd.DataFrame(text_chunks, columns=["chunk"])
			data_df["book"] = book_name
			print("Shape:", data_df.shape)
			print(data_df.head())

			jsonl_filename = os.path.join(OUTPUT_FOLDER, f"chunks-{method}-{book_name}.jsonl")
			with open(jsonl_filename, "w") as json_file:
				json_file.write(data_df.to_json(orient='records', lines=True))


def embed(method="char-split", max_chunks=None):
	print("embed()")

	jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"chunks-{method}-*.jsonl"))
	print("Number of files to process:", len(jsonl_files))

	for jsonl_file in jsonl_files:
		print("Processing file:", jsonl_file)

		data_df = pd.read_json(jsonl_file, lines=True)

		if max_chunks:
			data_df = data_df.head(max_chunks)  

		print("Shape:", data_df.shape)
		print(data_df.head())

		chunks = data_df["chunk"].values
		embeddings = generate_text_embeddings(chunks, EMBEDDING_DIMENSION, batch_size=100)
		data_df["embedding"] = embeddings

		time.sleep(5)

		print("Shape:", data_df.shape)
		print(data_df.head())

		jsonl_filename = jsonl_file.replace("chunks-", "embeddings-")
		with open(jsonl_filename, "w") as json_file:
			json_file.write(data_df.to_json(orient='records', lines=True))



def load(method="char-split"):
	print("load()")

	# Clear Cache
	chromadb.api.client.SharedSystemClient.clear_system_cache()

	# Connect to chroma DB
	client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

	# Get a collection object from an existing collection, by name. If it doesn't exist, create it.
	collection_name = f"{method}-collection"
	print("Creating collection:", collection_name)

	try:
		# Clear out any existing items in the collection
		client.delete_collection(name=collection_name)
		print(f"Deleted existing collection '{collection_name}'")
	except Exception:
		print(f"Collection '{collection_name}' did not exist. Creating new.")

	collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})
	print(f"Created new empty collection '{collection_name}'")
	print("Collection:", collection)

	# Get the list of embedding files
	jsonl_files = glob.glob(os.path.join(OUTPUT_FOLDER, f"embeddings-{method}-*.jsonl"))
	print("Number of files to process:", len(jsonl_files))

	# Process
	for jsonl_file in jsonl_files:
		print("Processing file:", jsonl_file)

		data_df = pd.read_json(jsonl_file, lines=True)
		print("Shape:", data_df.shape)
		print(data_df.head())

		# Load data
		load_text_embeddings(data_df, collection)

# def chat(question, method="char-split"):
def chat(question, method="recursive-split"):
	
	client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
	collection_name = f"{method}-collection"
	query_embedding = generate_query_embedding(question)
	collection = client.get_collection(name=collection_name)
	results = collection.query(
		query_embeddings=[query_embedding],
		n_results=10
	)

	# INPUT_PROMPT = f"""
	# {question}
	# {"\n".join(results["documents"][0])}
	# """

	docs_text = "\n".join(results["documents"][0])
	INPUT_PROMPT = f"""
	{question}
	{docs_text}
	"""
	
	response = generative_model.generate_content(
		[INPUT_PROMPT],  # Input prompt
		generation_config=generation_config,  # Configuration settings
		stream=False,  # Enable streaming for responses
	)
	generated_text = response.text.strip()
	print("RAG Response:", generated_text)
	return generated_text

def main(args=None):
	print("CLI Arguments:", args)

	if args.chunk:
		chunk(method=args.chunk_type)

	if args.embed:
		embed(method=args.chunk_type)

	if args.load:
		load(method=args.chunk_type)

	if args.chat:
		question = input("Ask question: ")
		chat(question= question, method=args.chunk_type)
	
	if args.download:
		download()
	
	if args.run_pipeline:
		download()
		chunk(method=args.chunk_type)
		embed(method=args.chunk_type)
		load(method=args.chunk_type)
	if args.test:
		print("testing docker exec")

#### FastAPI route for inference
class RAGRequest(BaseModel):
    question: str
    method: Literal["char-split", "recursive-split"] = "recursive-split"
# @app.on_event("startup")

@asynccontextmanager
async def load_pre_reqs(app: FastAPI):
    if os.environ.get("RAG_ENV") == "test":
        print("⚠️ RAG_ENV=test: Inserting dummy collection for test mode")
        from chromadb import HttpClient
        client = HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

        collection_name = "recursive-split-collection"
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = client.create_collection(name=collection_name)
        collection.add(
            ids=["1"],
            documents=["Stress is a common issue."],
            metadatas=[{"book": "Dummy"}],
            embeddings=[[0.1]*256]  # low-dim dummy vector
        )

        yield
        return

    # Regular pipeline
    download()
    chunk(method='recursive-split')
    embed(method='recursive-split')
    load(method='recursive-split')
    yield


app = FastAPI(lifespan=load_pre_reqs)


@app.post("/rag/str")
def rag_string(question: str):
	return chat(question=question)

@app.post("/rag")
def rag_json(req: RAGRequest):
    valid_methods = ["char-split", "recursive-split"]
    if req.method not in valid_methods:
        return {"error": f"Unsupported method '{req.method}'"}, 422

    return chat(question=req.question, method=req.method)

##### Main CLI with argparse
if __name__ == "__main__":
	# Generate the inputs arguments parser
	# if you type into the terminal '--help', it will provide the description
	parser = argparse.ArgumentParser(description="CLI")

	parser.add_argument(
        "--download",
        action="store_true",
        help="Download books from GCS bucket",
    )
	parser.add_argument(
		"--chunk",
		action="store_true",
		help="Chunk text",
	)
	parser.add_argument(
		"--embed",
		action="store_true",
		help="Generate embeddings",
	)
	parser.add_argument(
		"--load",
		action="store_true",
		help="Load embeddings to vector db",
	)
	parser.add_argument(
		"--chat",
		action="store_true",
		help="Chat with LLM",
	)
	parser.add_argument(
        "--run_pipeline",
        action="store_true",
        help="For any selected chunk type, execute the pipeline: Download -> Chunk -> Embed -> Load",
    )
	parser.add_argument(
        "--test",
        action="store_true",
        help="For any selected chunk type, execute the pipeline: Download -> Chunk -> Embed -> Load",
    )
	parser.add_argument("--chunk_type", default="recursive-split", help="char-split | recursive-split")

	args = parser.parse_args()
	main(args)
