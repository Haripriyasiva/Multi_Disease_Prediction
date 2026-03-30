
import os
import chromadb
from sentence_transformers import SentenceTransformer

def build_knowledge_base():
    # Initialize ChromaDB client with a persistent path
    client = chromadb.PersistentClient(path="chroma_db")

    # Create or get a collection
    collection = client.get_or_create_collection("medical_documents")

    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Path to the medical documents
    documents_path = 'medical_documents'
    documents = []
    metadatas = []
    ids = []
    id_counter = 1

    # Iterate over the directories (Heart, Kidney, Diabetes)
    for disease in os.listdir(documents_path):
        disease_path = os.path.join(documents_path, disease)
        if os.path.isdir(disease_path):
            # Iterate over the files in each directory
            for doc_file in os.listdir(disease_path):
                file_path = os.path.join(disease_path, doc_file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    metadatas.append({"disease": disease, "source": doc_file})
                    ids.append(str(id_counter))
                    id_counter += 1

    # Generate embeddings
    embeddings = model.encode(documents)

    # Add documents to the collection
    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print("Knowledge base built successfully.")

if __name__ == '__main__':
    build_knowledge_base()
