import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
import open_clip
from open_clip import create_model_and_transforms
import torch.nn.functional as F

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B-32"
pretrained = "openai"
embedding_file = "./image_embeddings.pickle"  # Path to precomputed embeddings
image_folder = "../coco_images_resized"  # Path to the image dataset folder

# Load the model and tokenizer
model, preprocess_train, preprocess_val = create_model_and_transforms(model_name, pretrained=pretrained)
model = model.to(device)
model.eval()
text_tokenizer = open_clip.get_tokenizer(model_name)


# Load precomputed embeddings
def load_image_embeddings():
    """Load precomputed image embeddings."""
    if not os.path.exists(embedding_file):
        raise FileNotFoundError(f"Embedding file {embedding_file} not found.")
    return pd.read_pickle(embedding_file)

# Compute cosine similarity
def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def image_to_image_query(query_image_path):
    """Perform Image-to-Image search and return top 5 results."""
    embeddings_df = load_image_embeddings()
    query_image = Image.open(query_image_path).convert("RGB")
    query_image_tensor = preprocess_val(query_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(query_image_tensor)
        image_embedding = F.normalize(image_embedding, p=2, dim=1).cpu().numpy()

    # Compute similarities for all images
    results = []
    for _, row in embeddings_df.iterrows():
        similarity = cosine_similarity(image_embedding[0], row["embedding"])
        results.append({"file_name": row["file_name"], "similarity": float(similarity)})

    # Sort by similarity and return top 5
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]
    return results


def text_to_image_query(query_text):
    """Perform Text-to-Image search and return top 5 results."""
    embeddings_df = load_image_embeddings()
    text_tokens = text_tokenizer([query_text]).to(device)

    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding = F.normalize(text_embedding, p=2, dim=1).cpu().numpy()

    # Compute similarities for all images
    results = []
    for _, row in embeddings_df.iterrows():
        similarity = cosine_similarity(text_embedding[0], row["embedding"])
        results.append({"file_name": row["file_name"], "similarity": float(similarity)})

    # Sort by similarity and return top 5
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]
    return results



def hybrid_query(query_image_path, query_text, lambda_weight):
    """Perform a hybrid search combining text and image, returning top 5 results."""
    embeddings_df = load_image_embeddings()

    # Process the image query
    query_image = Image.open(query_image_path).convert("RGB")
    query_image_tensor = preprocess_val(query_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(query_image_tensor)
        image_embedding = F.normalize(image_embedding, p=2, dim=1)  # Keep as PyTorch tensor

    # Process the text query
    text_tokens = text_tokenizer([query_text]).to(device)

    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)  # Keep as PyTorch tensor

    # Combine embeddings with lambda weight
    hybrid_embedding = F.normalize(
        lambda_weight * text_embedding + (1 - lambda_weight) * image_embedding, p=2, dim=1
    )

    # Convert the hybrid embedding to NumPy for similarity computation
    hybrid_embedding = hybrid_embedding.cpu().numpy()

    # Compute similarities for all images
    results = []
    for _, row in embeddings_df.iterrows():
        similarity = cosine_similarity(hybrid_embedding[0], row["embedding"])
        results.append({"file_name": row["file_name"], "similarity": float(similarity)})

    # Sort by similarity and return top 5
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]
    return results

