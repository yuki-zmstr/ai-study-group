import os
import numpy as np
from openai import OpenAI


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


def get_embedding(text, client, model="text-embedding-3-small"):
    """Get embedding for a text string from OpenAI API."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def main():
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Define two strings to compare
    string1 = "半蔵門への行き方を教えて"
    string2 = "半蔵門のおいしい中華"

    # Get embeddings for both strings
    embedding1 = get_embedding(string1, client)
    embedding2 = get_embedding(string2, client)

    # Calculate cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)

    print(f"String 1: {string1}")
    print(f"String 2: {string2}")
    print(f"Cosine Similarity: {similarity:.4f}")

    return similarity


if __name__ == "__main__":
    main()
