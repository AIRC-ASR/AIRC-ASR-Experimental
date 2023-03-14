import redis
import json
import pandas as pd
from torch.optim import Adam
from knowledge_graph import KB, from_text_to_kb
from torchkge.evaluation import LinkPredictionEvaluator
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import TransEModel
from torchkge.utils.datasets import load_fb15k
from torchkge.utils import Trainer, MarginLoss


with open("kg_plan_rewrite/redis_creds.json", "r") as file:
    creds = json.loads(file.read())
db = redis.Redis(**creds)
db.set("foo", "bar")
value = db.get('foo')
print(value)


def trained_embed(kg_train: KnowledgeGraph, kg_test: KnowledgeGraph, kg_val: KnowledgeGraph = None):
    # Define some hyper-parameters for training
    emb_dim = 100
    lr = 0.0004
    margin = 0.5
    n_epochs = 1000
    batch_size = 32768

    # Load dataset
    # kg_train, kg_val, kg_test = load_fb15k()

    # Define the model and criterion
    model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    criterion = MarginLoss(margin)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    trainer = Trainer(model, criterion, kg_train, n_epochs, batch_size,
                      optimizer=optimizer, sampling_type='bern', use_cuda='all',)

    trainer.run()

    evaluator = LinkPredictionEvaluator(model, kg_test)
    evaluator.evaluate(20)
    evaluator.print_results()
    return model


# TODO: Finish this function
def text_to_kg_embedding(train_texts: list[str], test_texts: list[str]):
    '''This function takes a piece of text, creates a knowledge graph
    from it based on the REBEL model and return knowledge graph embeddings.'''
    # https://pykg2vec.readthedocs.io/en/latest/intro.html
    # TODO: Check if the text is in a cache, and if so, return the knowledge graph
    # TODO: embeddings from the cache, otherwise, store it in the cache
    # TODO: Also store the exact model configurations
    # TODO: Optionally, add tuning of the model here
    print("Creating graphs...")
    dataset = (from_text_to_kb(train_texts[0]), from_text_to_kb(test_texts[0]))

    for i, texts in enumerate([train_texts, test_texts]):
        for text in texts[1:]:
            tmp = from_text_to_kb(text)
            dataset[i].combine(tmp)
        
    
    dataset = [kb.to_torch_kg() for kb in dataset]

    print("Training embeddings...")
    embed_model = trained_embed(*dataset)
    embeddings = embed_model.get_embeddings()
    return embeddings

if __name__ == "__main__":
    print(text_to_kg_embedding(["Barack Obama eats pasta with George Clooney"], ["Barack Obama eats pasta with George Clooney"]))
