import torch
import os
from sentence_transformers import SentenceTransformer, models

_model = None

def description_reward(pred_desc, true_desc):
    """
    Sentence-BERT
    """
    current_model = None
    if current_model is None:
        model_path = '/mnt/nfs/lyh/Project/IAD-R1/all-MiniLM-L6-v2'
        
        try:
            word_embedding_model = models.Transformer(model_path)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
            
            current_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        except Exception as e:
            print(f"error: {str(e)}")
            
            # Download from Hugging Face
            current_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("Loaded successfully !")
    
    # process
    pred_desc = pred_desc.lower().strip()
    true_desc = true_desc.lower().strip()
    
    # exact match
    if pred_desc == true_desc:
        return 1.0
    
    if not pred_desc or not true_desc:
        return 0.0
    
    # Calculate similarity
    try:

        pred_embedding = current_model.encode(pred_desc, convert_to_tensor=True)

        true_embedding = current_model.encode(true_desc, convert_to_tensor=True)

        print(f"pred_embedding shape: {pred_embedding.shape}")
        print(f"true_embedding shape: {true_embedding.shape}")
        similarity = torch.cosine_similarity(pred_embedding, true_embedding, dim=0).item()
        print(f"predict: {pred_desc} | gt: {true_desc} | sim: {similarity}")
    except Exception as e:
        print(f"error: {str(e)}")
        similarity = 0.0
    return similarity

