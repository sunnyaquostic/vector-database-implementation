# Vector Database Learning with Python

A hands-on Python project exploring **text embeddings, vector storage, and semantic search** using Google's Gemini Embedding Model and ChromaDB.

The goal of this project is to understand how text can be converted into numerical representations (embeddings), stored in a vector database, and retrieved based on semantic similarity rather than exact keyword matching.

## Project Overview

Traditional search often relies on matching exact words or phrases. Semantic search takes a different approach: it represents text as vectors so that content with related meanings can be identified, even when the wording differs.

This project explores that process by:

- Generating text embeddings with Google's Gemini Embedding Model.
- Storing and retrieving vectors using ChromaDB.
- Comparing text embeddings using cosine similarity.
- Reducing embedding dimensions with Principal Component Analysis (PCA) for 3D visualization.
- Performing semantic similarity searches against stored vectors.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Google Gemini Embedding Model | Converts text into vector representations |
| ChromaDB | Persistent vector storage and similarity search |
| NumPy | Numerical operations on vectors |
| Pandas | Working with tabular data |
| Scikit-learn | Cosine similarity and PCA |
| Matplotlib | Visualizing embeddings |

## Understanding the Workflow

The project explores two main stages:

### 1. Data ingestion

1. Prepare the text data.
2. Generate embeddings using the Gemini embedding model.
3. Store the generated vectors and their associated text in ChromaDB.
4. Make the stored vectors available for similarity search.

### 2. Query and retrieval

1. Provide a search query.
2. Convert the query into an embedding using the same embedding model.
3. Search the vector database for similar vectors.
4. Retrieve the corresponding text documents.

This allows a query such as `purpose in existence` to retrieve text with related meaning, even if it does not contain those exact words.

## Project Structure

```text
vector-database-learning/
│
├── app/
│   ├── openai/
│   │   ├── gemini.py
│   │   └── gemini_embedding.py
│   │
│   ├── config.py
│   ├── db.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── .env.sample
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python installed on your machine.
- A Google Gemini API key.
- Git installed if you intend to clone the repository.

### 1. Clone the repository

```bash
git clone https://github.com/sunnyaquostic/vector-database-implementation.git
```

Move into the project directory:

```bash
cd vector-database-learning
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.sample` as a starting point.

Add your Gemini API key and the model configuration:

```env
GENAI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_gemini_embedding_model
```

The current `app/config.py` also declares `GEMINI_NAME`, `GEMINI_PROJECT_NAME`, and `GEMINI_PROJECT_ID` as required settings. Provide these in `.env` as well, or make them optional in the configuration if they are not needed by the project.

**Important:** Never commit your `.env` file or expose your API key publicly.

### 5. Run the project

```bash
python main.py
```

The main script demonstrates embedding generation and semantic retrieval. For a meaningful search result, ensure that the relevant text embeddings have first been added to the ChromaDB collection.

## Concepts Explored

During this project, I explored several foundational concepts behind vector databases and semantic search:

- **Vectors:** Ordered collections of numerical values used to represent information.
- **Embeddings:** Numerical representations of data that capture semantic relationships.
- **Tokenization:** The process of breaking text into tokens before model processing.
- **Cosine similarity:** A measure of similarity based on the angle between two vectors.
- **Euclidean distance:** A measure of the straight-line distance between two vectors.
- **Dimensionality reduction:** Using PCA to project high-dimensional embeddings into fewer dimensions for visualization.
- **Semantic search:** Retrieving information based on meaning and context rather than exact keyword matches.

## Why This Project?

I wanted to move beyond simply using AI-powered tools and understand some of the underlying concepts that make semantic search and retrieval possible.

Building this project helped me connect the theory of embeddings and vector similarity with a practical Python implementation.

It is a learning project, and there is still room to improve and explore more advanced vector database concepts.

## Future Improvements

Some areas I would like to explore further include:

- Experimenting with different embedding models and task types.
- Exploring metadata filtering and more advanced retrieval strategies.
- Comparing different similarity metrics.
- Building a simple Retrieval-Augmented Generation (RAG) application.
- Improving error handling and project configuration.

## Acknowledgement

This project was developed as part of my ongoing learning journey in Python, machine learning, and generative AI.

**Learning by building, one concept at a time.**