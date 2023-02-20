import logging
import os
import torch
import math
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as functional
import numpy as np
from tensorboardX import SummaryWriter
from bi_training_core import compute_loss, compute_loss_ae, Device, top_k_top_p_filtering
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from tqdm import tqdm
from sklearn.manifold import TSNE


logger = logging.getLogger("transformers")
logger.setLevel(logging.INFO)

def validate_step(model, tokenizer, model_type, val_loader, num_iters, max_val_batches, loss_fn, save_folder):
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)

    model.eval()

    n_words_bpe = 0
    n_words = 0
    logp_sum = 0.0
    kl_loss_sum = 0.0

    logger.info("Validation loop.         Batches: %d" % len(val_loader))
    logger.info("Validation loop. max_val_batches: %d" % max_val_batches)

    # val_iter = iter(val_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(val_iter)
    with tqdm(total=min(len(val_loader), max_val_batches)) as pbar:
        for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(val_loader):
            with torch.no_grad():
                if model_type == 'cvae':
                    loss, ce_loss, kl_loss = compute_loss(model, x_mask, x_tokens, y_mask, y_tokens,
                                                            input_tokens, target_tokens, mask, loss_fn, 1.0)
                else:
                    loss, ce_loss, kl_loss = compute_loss_ae(model, x_mask, x_tokens, y_mask, y_tokens,
                                                            input_tokens, target_tokens, mask, loss_fn, 1.0)

            if len(target_tokens.size()) == 1:
                target_tokens = target_tokens.unsqueeze(0)
            n, l = target_tokens.size()

            text = target_tokens[0, :].tolist()
            logprob = ce_loss.tolist()
            assert len(text) == len(logprob)

            # only for story
            idx = text.index(endoftext)
            text = text[idx + 1:]
            logprob = logprob[idx + 1:]

            if endoftext in text:
                idx = text.index(endoftext)
                text = text[:idx]
                logprob = logprob[:idx]

            logp_sum += sum(logprob)

            n_words_bpe += len(text)

            story = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
            story = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in story]
            story = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in
                        story]
            words = sum([len(
                [t for t in re.split('("|\'|!|\?|\.|,|:| |\n|’|“|”|;|\(|\)|`)', s) if t != ' ' and t != '']) for
                s in story])
            n_words += words

            kl_loss_sum += kl_loss.item()

            if i > max_val_batches:
                break
            pbar.update(1)

    loss_bpe = logp_sum / n_words_bpe
    ppl_bpe = round(math.exp(min(logp_sum / n_words_bpe, 100)), 3)
    ppl_word = round(math.exp(min(logp_sum / n_words, 100)), 3)
    kl = kl_loss_sum / len(val_loader)

    v_writer.add_scalar('loss', loss_bpe, num_iters)
    v_writer.add_scalar('ppl_bpe', ppl_bpe, num_iters)
    v_writer.add_scalar('ppl_word', ppl_word, num_iters)
    v_writer.add_scalar('kl', kl, num_iters)
    logger.info('val loss    : %.4f' % loss_bpe)
    logger.info('val ppl_bpe : %.4f' % ppl_bpe)
    logger.info('val ppl_word: %.4f' % ppl_word)
    logger.info('val   kl    : %.4f' % kl)

    model.train()

def plot_input_distribution(model, tokenizer, model_type, test_loader, dataset, num_iters, save_folder):
    model.eval()

    # get embedding
    X_emb = None
    y = None

    # test_iter = iter(test_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(test_iter)
    with tqdm(total=len(test_loader)) as pbar:
        for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(test_loader):
            y_mask = y_mask.to(Device.device)
            y_tokens = y_tokens.to(Device.device)
            x_mask = x_mask.to(Device.device)
            x_tokens = x_tokens.to(Device.device)
            with torch.no_grad():
                if model_type == 'cvae':
                    latent_mean, latent_logvar = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
                else:
                    latent_mean, latent_logvar = model.encoder(input_ids=x_tokens, attention_mask=x_mask)[:2]

            if dataset == 'ax' or dataset == 'yp':
                label = [tokenizer.decode(l)[:2] for l in x_tokens.tolist()]
            elif dataset == 'wp':
                label = []
                prompts = [tokenizer.decode(l)[:6].lower() for l in x_tokens.tolist()]
                for prom in prompts:
                    if prom[0] in ['[', '('] and prom[5] in [']', ')']:
                        label.append(prom[2:4])
                    else:
                        label.append(None)
            elif dataset == 'wi':
                # 0. TV, play, miniseries, telenovela; 1.film; 2. music; 3. manga, comic, 4. book, novel, story 5. game
                label = []
                prompts = [tokenizer.decode(l) for l in x_tokens.tolist()]
                for prom in prompts:
                    if 'TV' in prom or 'play' in prom or 'miniseries' in prom or 'telenovela' in prom:
                        label.append(0)
                    elif 'film' in prom:
                        label.append(1)
                    elif 'music' in prom:
                        label.append(2)
                    elif 'manga' in prom or 'comic' in prom:
                        label.append(3)
                    elif 'book' in prom or 'novel' in prom or 'story' in prom:
                        label.append(4)
                    elif 'game' in prom:
                        label.append(5)
                    else:
                        label.append(None)
            else:
                raise Exception

            if i == 0:
                X_emb = latent_mean.data
                y = label
            else:
                X_emb = torch.cat((X_emb, latent_mean.data), dim=0)
                y.extend(label)
            pbar.update(1)
    X_emb = X_emb.cpu().numpy()

    try:
        if dataset == 'yp':
            y = ['0' if l in ['0', '1'] else l for l in y]
            y = ['4' if l in ['3', '4'] else l for l in y]
            X_emb = X_emb[[l != '2' for l in y], :]
            y = [l for l in y if l != '2']

        if dataset == 'wp':
            topics = [['wp', 'sp', 'tt'], ['eu'], ['cw'], ['pm'], ['mp', 'ip'], ['pi', 'cc'], ['ot'], ['rf']]
            match = [[True if l in t else False for t in topics] for l in y]
            y = [m.index(True) if True in m else None for m in match]
            X_emb = X_emb[[l is not None for l in y], :]
            y = [l for l in y if l is not None]

        if dataset == 'wi':
            X_emb = X_emb[[l is not None for l in y], :]
            y = [l for l in y if l is not None]

        # to 2D
        # X_emb_2d = TSNE(n_components=2, init='pca', verbose=1).fit_transform(X_emb)
        X_emb_2d = TSNE(n_components=2, verbose=1, perplexity=40).fit_transform(X_emb)

        def remove_outliers(data, r=2.0):
            outliers_data = abs(data - np.mean(data, axis=0)) >= r * np.std(data, axis=0)
            outliers = np.any(outliers_data, axis=1)
            keep = np.logical_not(outliers)
            return outliers, keep

        outliers, keep = remove_outliers(X_emb_2d)
        X_emb_2d = X_emb_2d[keep, :]
        y = [l for l, k in zip(y, keep.tolist()) if k]

        # plot
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_axes([0, 0, 1, 1])
        cc = ['r', 'b', 'g', 'y', 'k', 'c', 'm', 'tab:blue']
        for i, l in enumerate(sorted(set(y))):
            idx = [yl == l for yl in y]
            plt.scatter(X_emb_2d[idx, 0], X_emb_2d[idx, 1], c=cc[i], s=10, edgecolor='none', alpha=0.5)
        ax.axis('off')  # adding it will get no axis
        plt.savefig(os.path.join(save_folder, 'tSNE_' + '{:07d}'.format(num_iters) + '.png'))
        plt.close(fig)
    except:
        pass

    model.train()


def sample_sequence(model, length, batch_size=None, x_mask=None, x_tokens=None, y_mask=None, y_tokens=None,
                    temperature=1, top_k=100, top_p=0.95, sample=True, eos_token=None, model_type='cvae'):
    x_mask = x_mask.to(Device.device)
    x_tokens = x_tokens.to(Device.device)
    y_mask = y_mask.to(Device.device)
    y_tokens = y_tokens.to(Device.device)

    with torch.no_grad():
        if model_type == 'cvae':
            try:
                prior_mean, prior_logvar = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
            except:
                prior_mean = prior_logvar = torch.zeros([batch_size, model.config.n_embd], device=Device.device)
            latent_mean, latent_logvar = prior_mean, prior_logvar
            z = model.reparameterize(latent_mean, latent_logvar)
            assert not torch.isnan(z).any(), 'training get nan z'
        else:
            posterior_mean, posterior_logvar = model.encoder(input_ids=x_tokens, attention_mask=x_mask)[:2]
            latent_mean, latent_logvar = posterior_mean, posterior_logvar
            z = latent_mean
            assert not torch.isnan(z).any(), 'training get nan z'

        _, mem = model.transformer(input_ids=x_tokens[:, :-1], past=None, attention_mask=x_mask[:, :-1], representations=z)
        prev = x_tokens[:, -1].view(batch_size, -1)

        output = prev
        probability = torch.tensor([], dtype=z.dtype, device=Device.device)
        if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=Device.device)

        for i in range(length): #trange
            logits, mem = model.transformer(input_ids=prev, past=mem, representations=z)

            logits = model.lm_head(logits)
            if model.add_softmax:
                logits_rep = model.lm_head_rep(z)
                logits = logits + logits_rep.unsqueeze(dim=1)

            logits = logits[:, -1, :] / temperature
            logits = top_k_top_p_filtering(logits, top_k, top_p)
            probs = functional.softmax(logits, dim=-1)
            if sample:
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                _, next_token = torch.topk(probs, k=1, dim=-1)

            probability = torch.cat((probability, probs.gather(1, next_token)), dim=1)
            output = torch.cat((output, next_token), dim=1)
            prev = next_token

            # early stopping if all sents have ended once
            if_end[next_token.view(-1).eq(eos_token)] = True
            if if_end.all(): break
    return output, probability


def generate_samples(model, tokenizer, args, test_loader, num_iters, save_folder):
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    model.eval()

    n_samples = 0
    bleu4_sum = 0.0
    rouge_scores_values_sum = [0.0] * 9

    args.nsamples = 1
    args.batch_size = 1
    args.temperature = 0.95
    args.top_k = 100
    args.top_p = 0.95
    model_type = args.model_type

    # write samples to file
    samples_file = open(os.path.join(save_folder, 'generate-' + '%07d' % num_iters + '.txt'), 'w', encoding='utf8')

    # test_iter = iter(test_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(test_iter)
    with tqdm(total=len(test_loader)) as pbar:
        for i_test, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(
                test_loader):

            if i_test >= 10: break
            if n_samples == 0: break

            length = -1
            if length == -1:
                length = model.config.n_ctx - x_tokens.size(1) - 1
            elif length > model.config.n_ctx - x_tokens.size(1) - 1:
                raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

            eff_samples = []
            n, l = target_tokens.size()
            storys = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
            storys = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in storys]
            storys_str = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in
                            storys]

            for _ in range(args.nsamples // args.batch_size):
                # model, batch_size, temperature, top_k, top_p, eos_token, sample = model, args.batch_size, args.temperature, args.top_k, args.top_p, tokenizer.encoder['<|endoftext|>'], True
                out, _ = sample_sequence(
                    model=model,
                    length=length,
                    batch_size=args.batch_size,
                    x_mask=x_mask,
                    x_tokens=x_tokens,
                    y_mask=y_mask,
                    y_tokens=y_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    eos_token=tokenizer.encoder['<|endoftext|>'],
                    model_type=model_type
                )
                out = out.tolist()

                # extract story, check metrics
                for i in range(len(out)):
                    text = out[i]
                    text = text[text.index(endoftext) + 1:]

                    if endoftext in text:
                        idx = text.index(endoftext)
                        text = text[:idx]

                    text = tokenizer.decode(text).strip()

                    # score for one long text, higher than 0.075 usually means repetition
                    # rep_score = repeat_score(text.split(), ngram=[3, 4, 5, 6, 7, 8])
                    # if rep_score > 0.075:
                    #     # logger.info(rep_score)
                    #     continue

                    try:
                        # check bleu
                        bleu4 = sentence_bleu([storys_str[i].split()], text,
                                                smoothing_function=SmoothingFunction().method7)

                        # check rouge
                        rouge = Rouge()
                        rouge_scores = rouge.get_scores(text, storys_str[i])
                        rouge_scores_values = [v for k in rouge_scores[0].keys() for v in
                                                rouge_scores[0][k].values()]

                        bleu4_sum += bleu4
                        rouge_scores_values_sum = [v1 + v2 for v1, v2 in
                                                    zip(rouge_scores_values_sum, rouge_scores_values)]
                        n_samples += 1
                    except:
                        bleu4 = 0.0
                        rouge_scores = [{'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}}]

                    eff_samples.append((text, bleu4, rouge_scores))

                pbar.update(1)

            for i in range(len(eff_samples)):
                samples_file.write("=" * 50 + " SAMPLE " + str(i_test) + " " + "=" * 50)
                samples_file.write('\n' * 2)

                samples_file.write("=" * 40 + " Outlines  " + "=" * 40)
                samples_file.write('\n' * 2)
                samples_file.write(tokenizer.decode(x_tokens[i, :][x_mask[i, :] == 1].tolist()))
                samples_file.write('\n' * 2)
                samples_file.write("=" * 40 + " Story " + "=" * 40)
                samples_file.write('\n' * 2)
                samples_file.write(storys_str[i])
                samples_file.write('\n' * 2)

                samples_file.write("=" * 40 + " Generated " + "=" * 40)
                samples_file.write('\n' * 2)
                samples_file.write(eff_samples[i][0])
                samples_file.write('\n' * 4)
                samples_file.flush()

    logger.info("Test complete with %05d samples.", n_samples)
    logger.info("Iteration completed: %d" % num_iters)

    if n_samples != 0:
        bleu4 = round(bleu4_sum / n_samples, 3)
        rouge_scores_values = [round(r / n_samples, 3) for r in rouge_scores_values_sum]
        logger.info(' bleu-4: %f', bleu4)
        logger.info(' rouge : %s', str(rouge_scores_values))

        model.train()