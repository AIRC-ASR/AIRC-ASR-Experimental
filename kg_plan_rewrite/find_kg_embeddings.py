from knowledge_graph import KB, from_text_to_kb
from torchkge.models.translation import TransDModel

# TODO: Finish this function
def text_to_kg_embedding(text):
    '''This function takes a piece of text, creates a knowledge graph
    from it based on the REBEL model and return knowledge graph embeddings.'''
    # https://pykg2vec.readthedocs.io/en/latest/intro.html
    # TODO: Check if the text is in a cache, and if so, return the knowledge graph
    # TODO: embeddings from the cache, otherwise, store it in the cache
    # TODO: Also store the exact model configurations
    # TODO: Optionally, add tuning of the model here

    kb = from_text_to_kb(text)
    embed_model = TransDModel(2, 2, len(kb.entities), len(kb.relations))
    embeddings = embed_model.get_embeddings()
    return embeddings

if __name__ == "__main__":
    text_to_kg_embedding("Barack Obama eats pasta with George Clooney")
