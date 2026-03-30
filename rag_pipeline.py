
import chromadb
from sentence_transformers import SentenceTransformer

def query_knowledge_base(query, n_results=3):
    """
    Queries the knowledge base for the most relevant documents.
    """
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("medical_documents")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results['documents'][0]

def generate_health_plan(risk_category):
    """
    Generates a health plan for a given risk category using the RAG pipeline.
    """
    # 1. Query Formulation
    query = f"What are the lifestyle and dietary recommendations for {risk_category}?"

    # 2. Retrieval
    retrieved_docs = query_knowledge_base(query)

    # 3. Generation (Simplified for now, will use a proper language model later)
    health_plan = "Based on your risk profile, here are some recommendations:\n\n"
    for doc in retrieved_docs:
        health_plan += f"- {doc}\n"
    
    # Add the mandatory disclaimer
    health_plan += "\n\nThis is an AI-generated wellness plan based on your latest vitals. Always consult your primary care physician before making drastic dietary changes."

    return health_plan

if __name__ == '__main__':
    # Example usage
    risk_category = "High Fasting Glucose, BMI 28"
    health_plan = generate_health_plan(risk_category)
    print(health_plan)
