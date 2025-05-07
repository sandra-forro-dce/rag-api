import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


os.environ["RAG_ENV"] = "test"

from fastapi.testclient import TestClient
from rag import app  

client = TestClient(app)


def test_rag_str():
    response = client.post("/rag/str", params={"question": "What is anxiety?"})
    assert response.status_code == 200
    assert isinstance(response.text, str)
    assert len(response.text.strip()) > 0

def test_rag_json():
    payload = {
        "question": "How to deal with stress?",
        "method": "recursive-split"
    }
    response = client.post("/rag", json=payload)
    assert response.status_code == 200
    assert isinstance(response.text, str)
    assert len(response.text.strip()) > 0

def test_rag_json_with_invalid_method():
    payload = {
        "question": "How to sleep better?",
        "method": "unknown-method"
    }
    response = client.post("/rag", json=payload)
    assert response.status_code in [404, 500]

def test_rag_json_missing_question():
    payload = {
        "method": "recursive-split"
    }
    response = client.post("/rag", json=payload)
    assert response.status_code == 422



