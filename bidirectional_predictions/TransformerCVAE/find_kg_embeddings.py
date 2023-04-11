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
    # print(text_to_kg_embedding("Barack Obama eats pasta with George Clooney"))
    text = """
    Napoleon Bonaparte (born Napoleone di Buonaparte; 15 August 1769 – 5 May 1821), and later known by his regnal name Napoleon I, was a French military and political leader who rose to prominence during the French Revolution and led several successful campaigns during the Revolutionary Wars. He was the de facto leader of the French Republic as First Consul from 1799 to 1804. As Napoleon I, he was Emperor of the French from 1804 until 1814 and again in 1815. Napoleon's political and cultural legacy has endured, and he has been one of the most celebrated and controversial leaders in world history. Napoleon was born on the island of Corsica not long after its annexation by the Kingdom of France.[5] He supported the French Revolution in 1789 while serving in the French army, and tried to spread its ideals to his native Corsica. He rose rapidly in the Army after he saved the governing French Directory by firing on royalist insurgents. In 1796, he began a military campaign against the Austrians and their Italian allies, scoring decisive victories and becoming a national hero. Two years later, he led a military expedition to Egypt that served as a springboard to political power. He engineered a coup in November 1799 and became First Consul of the Republic. Differences with the British meant that the French faced the War of the Third Coalition by 1805. Napoleon shattered this coalition with victories in the Ulm Campaign, and at the Battle of Austerlitz, which led to the dissolving of the Holy Roman Empire. In 1806, the Fourth Coalition took up arms against him because Prussia became worried about growing French influence on the continent. Napoleon knocked out Prussia at the battles of Jena and Auerstedt, marched the Grande Armée into Eastern Europe, annihilating the Russians in June 1807 at Friedland, and forcing the defeated nations of the Fourth Coalition to accept the Treaties of Tilsit. Two years later, the Austrians challenged the French again during the War of the Fifth Coalition, but Napoleon solidified his grip over Europe after triumphing at the Battle of Wagram. Hoping to extend the Continental System, his embargo against Britain, Napoleon invaded the Iberian Peninsula and declared his brother Joseph King of Spain in 1808. The Spanish and the Portuguese revolted in the Peninsular War, culminating in defeat for Napoleon's marshals. Napoleon launched an invasion of Russia in the summer of 1812. The resulting campaign witnessed the catastrophic retreat of Napoleon's Grande Armée. In 1813, Prussia and Austria joined Russian forces in a Sixth Coalition against France. A chaotic military campaign resulted in a large coalition army defeating Napoleon at the Battle of Leipzig in October 1813. The coalition invaded France and captured Paris, forcing Napoleon to abdicate in April 1814. He was exiled to the island of Elba, between Corsica and Italy. In France, the Bourbons were restored to power. However, Napoleon escaped Elba in February 1815 and took control of France.[6][7] The Allies responded by forming a Seventh Coalition, which defeated Napoleon at the Battle of Waterloo in June 1815. The British exiled him to the remote island of Saint Helena in the Atlantic, where he died in 1821 at the age of 51. Napoleon had an extensive impact on the modern world, bringing liberal reforms to the many countries he conquered, especially the Low Countries, Switzerland, and parts of modern Italy and Germany. He implemented liberal policies in France and Western Europe.
    """

    kb = from_text_to_kb(text, verbose=True)
    kb.print()