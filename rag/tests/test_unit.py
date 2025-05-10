import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch, MagicMock
from rag import generate_query_embedding, generate_text_embeddings, chat

# Test generate_query_embedding 
@patch("rag.embedding_model")
def test_generate_query_embedding(mock_model):
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 256
    mock_model.get_embeddings.return_value = [mock_embedding]

    result = generate_query_embedding("What is anxiety?")
    assert isinstance(result, list)
    assert len(result) == 256
    assert result[0] == 0.1

# Test generate_text_embeddings 
@patch("rag.embedding_model")
def test_generate_text_embeddings_success(mock_model):
    mock_embedding = MagicMock()
    mock_embedding.values = [0.5] * 256
    mock_model.get_embeddings.side_effect = lambda inputs, **kwargs: [mock_embedding for _ in inputs]


    chunks = ["Chunk 1", "Chunk 2"]
    result = generate_text_embeddings(chunks, batch_size=1)

    assert len(result) == 2
    assert all(len(vec) == 256 for vec in result)

# Test chat function
@patch("rag.generative_model.generate_content")
@patch("rag.chromadb.HttpClient")
@patch("rag.generate_query_embedding")
def test_chat_success(mock_embed, mock_chroma_client, mock_gen):
    mock_embed.return_value = [0.1] * 256

    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["Stress can affect your mood."]]
    }

    mock_chroma_client.return_value.get_collection.return_value = fake_collection

    mock_gen.return_value.text = "Stress is a serious issue."

    result = chat("What is stress?")
    
    assert isinstance(result, str)
    assert "stress" in result.lower()
