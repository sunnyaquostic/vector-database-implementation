from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
import numpy as np

def create_data_frame(texts: list, result):
    for test, embedding_output in zip(texts, result.embeddings):
        embedding = embedding_output.values
        print(f"Text: {test}")
        print(f"Embedding len: {len(embedding)}")
        print(f"Embedding: {embedding}\n")
        print("_" * 50)
        
    df = pd.DataFrame(
        cosine_similarity([embedding.values for embedding in result.embeddings]),
        index=texts,
        columns=texts
    )

    display(df)
    
def create_plot_3d_table(texts: list, result):
    embeddings = np.array([embedding.values for embedding in result.embeddings])
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(embeddings)

    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y', 'z'])
    df['text'] = texts

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df['x'], df['y'], df['z'])

    for i, txt in enumerate(df['text']):
        ax.text(df['x'][i], df['y'][i], df['z'][i], txt, size=10, zorder=1, color='k')
        
    ax.set_title('3D PCA of Text Embeddings')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')

    plt.show()
    
def get_suggestion_from_database(*, db, client, query_text: str = "programming") -> list:
    query_text = query_text
    query_embedding = client.retrieve_embeddings(query_text=query_text)
    results = db.read(query_embedding=query_embedding)
    
    return results

def add_raw_vector_to_terminar(result, db, texts):
    embeddings = []
    ids = []

    for i, embedding_output in enumerate(result.embeddings):
        embeddings.append(embedding_output.values)
        ids.append(f"embedding_{i}")

    
    res = db.add(ids=ids, embeddings=embeddings, texts=texts) 
    
    return res
