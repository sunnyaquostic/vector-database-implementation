
from app.openai.gemini_embedding import EmbeddingClient
from app.db import Database
from app.utils import create_data_frame, create_plot_3d_table, add_raw_vector_to_terminar, get_suggestion_from_database

TEXTS = [
    "What is the meaning of life?",
    "What is the purpose of existence?",
    "How can I find happiness?",
    "What is the nature of reality?",
    "What is the ultimate truth?",
    "How do I write a function in python?",
    "How can I create a class in Java?",
    "Which smartphone has the best camera in 2024?",
    "Turn off the lights in the living room",
    "Apple released a new iPhone model in 2024 with improved camera features.",
    "Apple is my favorite fruit."
]

QUERY_TEXT = "purpose in existence"

DB_DIR = "chroma_db"
COLLECTION_NAME = "sunny-chroma-db-demo"

# Enable or disable visualizations
SHOW_SIMILARITY_MATRIX = True
SHOW_3D_VISUALIZATION = True


# Initialize clients
def initialize_services():

    embedding_client = EmbeddingClient()
    database = Database(db_dir=DB_DIR, name=COLLECTION_NAME)

    return embedding_client, database


# Generate embeddings
def generate_embeddings(client, texts):
    print("\nGenerating embeddings...")

    result = client.create_embeddings(texts=texts)

    if not result or not result.embeddings:
        raise RuntimeError("Embedding generation failed.")

    print(f"Successfully generated {len(result.embeddings)} embeddings.")

    print(
        f"Embedding dimensions: "
        f"{len(result.embeddings[0].values)}"
    )

    return result


# Explore embedding similarity

def explore_embeddings(texts, result):
    if SHOW_SIMILARITY_MATRIX:
        print("\nGenerating cosine similarity matrix...")

        create_data_frame(texts=texts, result=result)

    if SHOW_3D_VISUALIZATION:
        print("\nGenerating 3D embedding visualization...")

        create_plot_3d_table(texts=texts, result=result)


# 5. Store embeddings in ChromaDB
def store_embeddings(result, database, texts):

    print("\nStoring embeddings in ChromaDB...")
    response = add_raw_vector_to_terminar(result=result, db=database, texts=texts)

    print(response)


# Perform semantic search
def perform_semantic_search(client, database, query_text):

    print(f"\nSearching for: '{query_text}'")
    results = get_suggestion_from_database(db=database, client=client, query_text=query_text)

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    print("\nMost relevant documents:")

    for index, document in enumerate(documents):

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        print(f"\n{index + 1}. {document}")

        if distance is not None:
            print(f"   Distance: {distance:.4f}")


# 7. Application entry point
def main():

    print("=" * 55)
    print("VECTOR DATABASE LEARNING PROJECT")
    print("=" * 55)

    client, database = initialize_services()

    # Generate embeddings
    embedding_result = generate_embeddings(client=client, texts=TEXTS)

    # Explore vector representations
    explore_embeddings(texts=TEXTS, result=embedding_result)

    # Store vectors
    store_embeddings(result=embedding_result, database=database, texts=TEXTS)

    # Retrieve semantically similar documents
    perform_semantic_search(client=client, database=database, query_text=QUERY_TEXT)

    print("\n" + "=" * 55)
    print("DEMONSTRATION COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
