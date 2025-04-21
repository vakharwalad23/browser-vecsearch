import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from datasets import load_dataset
import onnx
import onnxruntime as ort
import numpy as np
from torch.utils.data import DataLoader, Subset
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
from torch.nn import CosineEmbeddingLoss

# Define teacher and student models
teacher_model_name = "sentence-transformers/all-mpnet-base-v2"
teacher_model = AutoModel.from_pretrained(teacher_model_name)
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

student_model_name = "sentence-transformers/all-MiniLM-L12-v2"
student_transformer = AutoModel.from_pretrained(student_model_name)
projection = nn.Linear(student_transformer.config.hidden_size, teacher_model.config.hidden_size)

class StudentModel(nn.Module):
    def __init__(self, transformer, projection):
        super().__init__()
        self.transformer = transformer
        self.projection = projection
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        projected_embeddings = self.projection(embeddings)
        return projected_embeddings

student_model = StudentModel(student_transformer, projection)

# Load and prepare dataset (Wikipedia)
dataset = load_dataset("wikipedia", "20220301.en", split="train[:100000]")
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

def tokenize_function(examples):
    return teacher_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train_dataset = train_dataset.dataset.map(tokenize_function, batched=True)
tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
train_dataloader = DataLoader(tokenized_train_dataset, batch_size=64, shuffle=True)

tokenized_val_dataset = val_dataset.dataset.map(tokenize_function, batched=True)
tokenized_val_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
val_dataloader = DataLoader(tokenized_val_dataset, batch_size=64)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
student_model.to(device)
teacher_model.eval()
student_model.train()

# Training setup
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
cosine_loss = CosineEmbeddingLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

num_epochs = 15
patience = 3
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    print(f"Starting Epoch {epoch+1}/{num_epochs}")
    # Training
    student_model.train()
    for batch in train_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
            teacher_embeddings = teacher_outputs.last_hidden_state.mean(dim=1)
        student_embeddings = student_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        target = torch.ones(inputs["input_ids"].size(0), device=device)
        loss = cosine_loss(student_embeddings, teacher_embeddings, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    student_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            teacher_outputs = teacher_model(**inputs)
            teacher_embeddings = teacher_outputs.last_hidden_state.mean(dim=1)
            student_embeddings = student_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            target = torch.ones(inputs["input_ids"].size(0), device=device)
            val_loss += cosine_loss(student_embeddings, teacher_embeddings, target).item()
    val_loss /= len(val_dataloader)

    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    # Adjust learning rate
    scheduler.step(val_loss)

# Define model for ONNX export
class EmbeddingsModel(nn.Module):
    def __init__(self, transformer, projection):
        super().__init__()
        self.transformer = transformer
        self.projection = projection
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        embeddings = hidden_states.mean(dim=1)
        projected_embeddings = self.projection(embeddings)
        return projected_embeddings

student_embeddings_model = EmbeddingsModel(student_transformer, projection)

# Prepare dummy input for export
dummy_input = teacher_tokenizer("This is a test sentence.", return_tensors="pt", padding="max_length", max_length=128)
dummy_input = {k: v.to("cpu") for k, v in dummy_input.items()}

# Export to ONNX
student_embeddings_model.to("cpu")
student_embeddings_model.eval()
torch.onnx.export(
    student_embeddings_model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "student_embeddings_model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["embeddings"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "embeddings": {0: "batch_size"}
    },
    opset_version=14
)

# Quantize the model
quantize_dynamic(
    "student_embeddings_model.onnx",
    "student_embeddings_model_quantized.onnx",
    weight_type=QuantType.QInt8
)

# Verify model size
print(f"Quantized model size: {os.path.getsize('student_embeddings_model_quantized.onnx') / 1e6} MB")

# Validate quantized model
ort_session = ort.InferenceSession("student_embeddings_model_quantized.onnx")
input_ids = dummy_input["input_ids"].numpy()
attention_mask = dummy_input["attention_mask"].numpy()
ort_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
ort_outputs = ort_session.run(None, ort_inputs)
onnx_embeddings = ort_outputs[0]

with torch.no_grad():
    pytorch_embeddings = student_embeddings_model(**dummy_input).numpy()

print("Difference between ONNX and PyTorch embeddings:", np.linalg.norm(onnx_embeddings - pytorch_embeddings))

# Save the quantized model
print("Quantized model saved as 'student_embeddings_model_quantized.onnx'")