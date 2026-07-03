import numpy as np

from app.services.embedding import get_embedding_service

def main() -> None:
    """ 
    Load the configured embedding model and verify
    document-query similarity.
    """
    
    documents  = [
        (
            "BERT is pretrained using masked "
            "language modelling and next sentence "
            "prediction."
        ), 
        (
             "PostgreSQL is a relational database "
            "used for durable application data."
        ),
        (
            "Qdrant stores vectors and supports "
            "semantic nearest-neighbour search."
        ),    
    ]
    
    query = (
        "How is BERT pretrained?"
    )
    
    service = get_embedding_service()

    print()
    print("EvidenceVault embedding verification.")
    print("="*45)
    print(f"Model: {service.model_name}")
    print(f"Configured dimension: {service.dimension}")
    print("Loading model and encoding text...")
    print()
    
    document_result = (
        service.embed_documents(documents)
    )
    
    query_result = (
        service.embed_query(query)
    )

    scores = (
        document_result.vectors @ query_result.vector
    )

    ranked_indexes = np.argsort(scores[::-1])
    
    print(f"Document embedding shape: {document_result.vectors.shape}")
    
    print(f"Query embedding shape: {query_result.vector.shape}")
    
    print()
    
    print("vector norms")
    print("="*45)
    
    for index, vector in enumerate(document_result.vectors):
        print(
            f"Document {index}: "
            f"{np.linalg.norm(vector):.6f}"
        )
        
        print(f"Query: {np.linalg.norm(query_result.vector):.6f}")
        
        print()
        print("Similarity ranking")
        print("-" * 45)
        
        for rank, document_index in enumerate(ranked_indexes, start=1):
            print(
                f"{rank}. Score "
                f"{scores[document_index]:.6f}"
            )
            print(
                f"   {documents[document_index]}"
            )
            
        best_document_index = int(ranked_indexes[0])
        
        print()
        print("Best matching document")
        print("-" * 45)
        
        print(documents[best_document_index])
        
if __name__ == "__main__":
    main()