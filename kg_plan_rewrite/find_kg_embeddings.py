from torch.optim import Adam
from knowledge_graph import from_text_to_kb
from torchkge.data_structures import KnowledgeGraph
from torchkge.models import TransEModel
from torchkge.utils import MarginLoss, DataLoader
from tqdm.autonotebook import tqdm
from torch import cuda
from torchkge.sampling import BernoulliNegativeSampler


def trained_embed(kg_train: KnowledgeGraph):
    # Define some hyper-parameters for training
    emb_dim = 100
    lr = 0.0004
    margin = 0.5
    n_epochs = 1000
    batch_size = 32768

    # Define the model and criterion
    print('n_ent', kg_train.n_ent, 'n_rel', kg_train.n_rel)
    model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
    criterion = MarginLoss(margin)

    # Move everything to CUDA if available
    if cuda.is_available():
        cuda.empty_cache()
        model.cuda()
        criterion.cuda()

    # Define the torch optimizer to be used
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    sampler = BernoulliNegativeSampler(kg_train)
    sampler.bern_probs = sampler.bern_probs.cuda()
    dataloader = DataLoader(kg_train, batch_size=batch_size, use_cuda='all')

    iterator = tqdm(range(n_epochs), unit='epoch')
    for epoch in iterator:
        running_loss = 0.0
        for batch in dataloader:
            h, t, r = batch[0], batch[1], batch[2]
            n_h, n_t = sampler.corrupt_batch(h, t, r)

            optimizer.zero_grad()

            # forward + backward + optimize
            pos, neg = model(h, t, r, n_h, n_t)
            loss = criterion(pos, neg)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        iterator.set_description(
            'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                                running_loss / len(dataloader)))

    model.normalize_parameters()
    return model


def text_to_kg_embedding(train_text: str):
    '''This function takes a piece of text, creates a knowledge graph
    from it based on the REBEL model and return knowledge graph embeddings.'''
    print("Creating graph...")
    kb = from_text_to_kb(train_text)
    torch_kb = kb.to_torch_kg()

    print("Training embeddings...")
    embed_model = trained_embed(torch_kb)
    embeddings = embed_model.get_embeddings()

    return embeddings


if __name__ == "__main__":
    print(text_to_kg_embedding("Barack Obama eats pasta with George Clooney"))
