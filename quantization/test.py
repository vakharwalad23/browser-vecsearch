import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Load the models and tokenizer
original_onnx_path = "student_embeddings_model.onnx"
quantized_onnx_path = "student_embeddings_model_quantized.onnx"
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# Create ONNX Runtime sessions
original_session = ort.InferenceSession(original_onnx_path)
quantized_session = ort.InferenceSession(quantized_onnx_path)

# Function to generate embeddings
def get_embeddings(session, input_text):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="np", padding="max_length", max_length=128)
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)
    
    # Run inference
    ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    ort_outputs = session.run(None, ort_inputs)
    return ort_outputs[0][0]  # Assuming single embedding output

# Sample input text
input_text = "This is a test sentence."

# Get embeddings from both models
original_embedding = get_embeddings(original_session, input_text)
quantized_embedding = get_embeddings(quantized_session, input_text)

# Function to compute cosine similarity
def cosine_similarity(emb1, emb2):
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    
    if norm_emb1 == 0 or norm_emb2 == 0:
        return 0
    
    emb1_normalized = emb1 / norm_emb1
    emb2_normalized = emb2 / norm_emb2
    return np.dot(emb1_normalized, emb2_normalized)

# Calculate and print cosine similarity
similarity = cosine_similarity(original_embedding, quantized_embedding)
print("Cosine Similarity between original and quantized embeddings:", similarity)