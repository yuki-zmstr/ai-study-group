import os
import numpy as np
from openai import OpenAI
from embedding_similarity import cosine_similarity, get_embedding


def simple_rag(query, texts, client, model="gpt-4o-mini"):
    """
    Simple RAG function that searches through texts, computes cosine similarity,
    and appends the most relevant text to the prompt before calling OpenAI.

    Args:
        query: The user's question/query
        texts: List of text documents to search through
        client: OpenAI client instance
        model: The chat model to use

    Returns:
        The assistant's response
    """
    # Get embedding for the query
    query_embedding = get_embedding(query, client)

    # Compute cosine similarity for each text
    similarities = []
    for text in texts:
        text_embedding = get_embedding(text, client)
        similarity = cosine_similarity(query_embedding, text_embedding)
        similarities.append(similarity)

    # Find the most relevant text
    best_idx = np.argmax(similarities)
    best_text = texts[best_idx]
    best_similarity = similarities[best_idx]

    print(f"\nMost relevant text (similarity: {best_similarity:.4f}):")
    print(f"{best_text}\n")

    # Create prompt with retrieved context
    augmented_prompt = f"""Use the following context to answer the question:

Context: {best_text}

Question: {query}

Answer:"""

    # Make chat completion request
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": augmented_prompt}
        ]
    )

    return response.choices[0].message.content


def main():
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Example: Two texts in our knowledge base
    texts = [
        "XAIONメンバーの平本さん、李さんはカラオケを特技としています。"
        "最近XAIONに入社された金田さんはバスケを特技としています。"
    ]

    # User query
    query = "XAION DATAでカラオケが上手がメンバーは誰？"

    # Run RAG
    answer = simple_rag(query, texts, client)

    print("Answer:")
    print(answer)


if __name__ == "__main__":
    main()
