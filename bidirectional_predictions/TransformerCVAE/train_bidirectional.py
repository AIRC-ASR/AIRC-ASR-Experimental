import os, time, gc, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, Conv1D
from tensorboardX import SummaryWriter
from tqdm import tqdm
from transformers.utils import logging
logger = logging.get_logger("transformers")
import copy
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from data.util import *
from util import *

from model import *

from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nltk

from bi_training_core import compute_loss, compute_loss_ae, train_step, top_k_top_p_filtering, Device
from bi_loss import bidirectional_loss

nltk.download('punkt')
nltk.download('stopwords')
# devices = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = devices

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
            probs = F.softmax(logits, dim=-1)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)

    # Default parameters are set based on single GPU training
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument('--data_type', type=str, default='t1', choices=['t' + str(i) for i in range(9)], help="t: type")
    parser.add_argument('--model_type', type=str, default='cvae', choices=['cvae', 'ae_vae_fusion'])
    parser.add_argument('--iterations', type=int, default=101640 * 4)  # wp 850001  wi 300001 ax 300001 yp 800001
    parser.add_argument('--dataset', type=str, default='wi', choices=['ax', 'yp', 'wp', 'wi'], help="Dataset to use for training")
    parser.add_argument('--warmup', type=int, default=10000,
                        help="Amount of iterations to warmup, then decay. (-1 for no warmup and decay)")

    parser.add_argument('--switch-time', type=float, default=0,
                        help="Percentage of iterations to spend on short sequence training.")
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out-dir', type=str, default='out')
    parser.add_argument('--load', type=str, help='path to load model from') # , default='out/test/'
    parser.add_argument('--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers')
    # use GPU
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--no_gpu', action="store_true")

    parser.add_argument('--fp16', action='store_true', help="Train using FP16?")
    parser.add_argument('--fp16_opt_level', default='O0', type=str, required=False)

    # KL cost annealing, increase beta from beta_0 to 1 in beta_warmup steps
    parser.add_argument('--beta_0', default=1.00, type=float)
    parser.add_argument('--beta_warmup', type=int, default=50000)
    # cyc_vae parameters
    parser.add_argument('--cycle', type=int, default=101640)

    parser.add_argument('--add_input', action="store_true")
    parser.add_argument('--add_attn', action="store_true")
    parser.add_argument('--add_softmax', action="store_true")
    parser.add_argument('--attn_proj_vary', action="store_true")

    parser.add_argument('--learn_prior', action="store_true")

    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)

    parser.add_argument('--short_seq_len', type=int, default=512)
    parser.add_argument('--long_seq_len', type=int, default=1024)

    # NOTE: Use for changing the arguments of the program
    args = parser.parse_args('test --add_input --learn_prior --fp16 --iterations 1000 --switch-time 0.5 '
                             '--train_batch_size 2 --val_batch_size 1 --test_batch_size 8 '
                             '--short_seq_len 512 --long_seq_len 1024 '.split()) # wi.12.proj_vary_beta_cvae

    if args.model_type == 'cvae':
        args.learn_prior = True
    else:
        args.learn_prior = False

    devices = '0'

    # GPU
    if not torch.cuda.is_available():
        args.no_gpu = True

    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        # print('Setting GPUs {}'.format(args.device))
        print('Using GPU devices {}'.format(devices))
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))

    Device.set_device(devices, args.gpu if gpu else "cpu")
    # device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    # logging
    save_folder = os.path.join(args.out_dir, args.experiment)
    os.makedirs(save_folder, exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
    # importlib.reload(logger)
    # logger.basicConfig(filename=os.path.join(save_folder, 'train.log'),
    #                     level=logger.INFO, format='%(asctime)s--- %(message)s')
    logger.info('\n*******************************************************************************\n')
    logger.info("the configuration:")
    logger.info(str(args).replace(',', '\n'))

    print('Loading models...')
    cache_dir = os.path.join(args.out_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    # Hack to allow tokenizing longer sequences.
    tokenizer.max_len = int(1e12)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
    print('gpt2_params:', num_params(gpt2_model))  # gpt2: 124439808
    config = GPT2Config()
    config.n_ctx = 1024

    # add special tokens
    # special_tokens_dict = {
    #     'pad_token': '<|startoftext|>',
    #     'cls_token': '<|startofcond|>',
    #     'sep_token': '<|sepofcond|>',
    #     'mask_token': '<|endofcond|>'
    # }
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print('We have added', num_added_toks, 'special tokens')
    # # Notice: resize_token_embeddings expect to receive the full size of the new vocab
    # gpt2_model.resize_token_embeddings(len(tokenizer))
    # assert tokenizer.pad_token == '<|startoftext|>'

    VAE = VAEModel(config, add_input=args.add_input, add_attn=args.add_attn, add_softmax=args.add_softmax,
                   attn_proj_vary=args.attn_proj_vary, learn_prior=args.learn_prior)
    init_para_frompretrained(VAE.transformer, gpt2_model.transformer, share_para=True)
    init_para_frompretrained(VAE.encoder, gpt2_model.transformer, share_para=False)
    if args.learn_prior:
        init_para_frompretrained(VAE.encoder_prior, VAE.encoder, share_para=True)
        VAE.encoder_prior.averageSelfAttention.attention_weights = VAE.encoder.averageSelfAttention.attention_weights
        
    VAE.lm_head.weight = gpt2_model.lm_head.weight
    if VAE.add_softmax:
        VAE.lm_head_rep = Conv1D(*gpt2_model.lm_head.weight.size())
        # VAE.lm_head_rep = LM_head_rep(*gpt2_model.lm_head.weight.size()[::-1])
    print('VAE_params:', num_params(VAE))  # 286694400
    if args.load:
        print('Loading model weights...')
        state = torch.load(os.path.join(args.load, 'model_latest.pt'))  # , map_location='cpu' model_latest.pt
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        VAE.load_state_dict(state)
        gc.collect()
    print('Done.')

    # fix pre-trained parameters before certain iterations
    tuning_all_after_iters = 40000
    tuning_all = False
    for name, parameter in VAE.named_parameters():
        # print((name, parameter.requires_grad))
        new_pars = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj', 'attn_proj', 'Nu_fc1', 'Nu_fc2', 'lm_head_rep']

        if not any([True if n in name else False for n in new_pars]):
           parameter.requires_grad = False

    print('Setup data...')
    curr_seq_len = args.short_seq_len
    train_loader, val_loader, test_loader = prepare_dataset(
        args.data_dir, args.dataset, tokenizer,
        args.train_batch_size, curr_seq_len,
        args.val_batch_size, curr_seq_len,
        args.test_batch_size, curr_seq_len,
        make_test=True,
        num_workers=args.workers, data_type=args.data_type
    )
    print('Done.')

    ###
    # val_loader = test_loader
    ###

    print('Wrapping models and optimizers...')

    # Apply linear scaling rule to increase batch size for short sequence training.
    curr_batch_size = args.train_batch_size
    curr_seq_len = args.short_seq_len
    lr_schedule = switch_schedule(linear_schedule(args), curr_batch_size / curr_seq_len,
                                  int(args.iterations * args.switch_time))
    VAE = VAE.to(Device.device)
    VAE.train()

    optimizer = torch.optim.AdamW(VAE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    # VAE, optimizer = amp.initialize(VAE, optimizer, opt_level=args.fp16_opt_level)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    print('Done.')

    print('Begin training iterations')
    logger.info("Begin training iterations")
    max_val_batches = 20000  # max num. of val batches
    logger.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    optimizer.zero_grad()
    beta = args.beta_0
    endoftext = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def val_step(val_loader):
        VAE.eval()

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
                    if args.model_type == 'cvae':
                        loss, ce_loss, kl_loss = compute_loss(VAE, x_mask, x_tokens, y_mask, y_tokens,
                                                              input_tokens, target_tokens, mask, loss_fn, 1.0)
                    else:
                        loss, ce_loss, kl_loss = compute_loss_ae(VAE, x_mask, x_tokens, y_mask, y_tokens,
                                                              input_tokens, target_tokens, mask, loss_fn, 1.0)

                if len(target_tokens.size()) == 1:
                    target_tokens = target_tokens.unsqueeze(0)
                n, l = target_tokens.size()

                text = target_tokens[0, :].tolist()
                logprob = ce_loss.tolist()

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

        VAE.train()

    def plot_input_distribution(test_loader, num_iters):
        VAE.eval()

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
                    if args.model_type == 'cvae':
                        latent_mean, latent_logvar = VAE.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
                    else:
                        latent_mean, latent_logvar = VAE.encoder(input_ids=x_tokens, attention_mask=x_mask)[:2]

                if args.dataset == 'ax' or args.dataset == 'yp':
                    label = [tokenizer.decode(l)[:2] for l in x_tokens.tolist()]
                elif args.dataset == 'wp':
                    label = []
                    prompts = [tokenizer.decode(l)[:6].lower() for l in x_tokens.tolist()]
                    for prom in prompts:
                        if prom[0] in ['[', '('] and prom[5] in [']', ')']:
                            label.append(prom[2:4])
                        else:
                            label.append(None)
                elif args.dataset == 'wi':
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
            if args.dataset == 'yp':
                y = ['0' if l in ['0', '1'] else l for l in y]
                y = ['4' if l in ['3', '4'] else l for l in y]
                X_emb = X_emb[[l != '2' for l in y], :]
                y = [l for l in y if l != '2']

            if args.dataset == 'wp':
                topics = [['wp', 'sp', 'tt'], ['eu'], ['cw'], ['pm'], ['mp', 'ip'], ['pi', 'cc'], ['ot'], ['rf']]
                match = [[True if l in t else False for t in topics] for l in y]
                y = [m.index(True) if True in m else None for m in match]
                X_emb = X_emb[[l is not None for l in y], :]
                y = [l for l in y if l is not None]

            if args.dataset == 'wi':
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

        VAE.train()

    def generate(test_loader, num_iters):
        VAE.eval()

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
                    length = VAE.config.n_ctx - x_tokens.size(1) - 1
                elif length > VAE.config.n_ctx - x_tokens.size(1) - 1:
                    raise ValueError("Can't get samples longer than window size: %s" % VAE.config.n_ctx)

                eff_samples = []
                n, l = target_tokens.size()
                storys = [tokenizer.decode(target_tokens[i, :]) for i in range(n)]
                storys = [s[s.find("<|endoftext|>") + len("<|endoftext|>"):] for s in storys]
                storys_str = [s[:s.find("<|endoftext|>") + len("<|endoftext|>")] if "<|endoftext|>" in s else s for s in
                              storys]

                for _ in range(args.nsamples // args.batch_size):
                    # model, batch_size, temperature, top_k, top_p, eos_token, sample = VAE, args.batch_size, args.temperature, args.top_k, args.top_p, tokenizer.encoder['<|endoftext|>'], True
                    out, _ = sample_sequence(
                        model=VAE,
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
                        #     # print(rep_score)
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

        print('Test complete with %05d samples.' % n_samples)
        logger.info("Test complete with %05d samples.", n_samples)
        logger.info("Iteration completed: %d" % num_iters)

        if n_samples != 0:
            bleu4 = round(bleu4_sum / n_samples, 3)
            rouge_scores_values = [round(r / n_samples, 3) for r in rouge_scores_values_sum]
            print(' bleu-4:', bleu4)
            print(' rouge :', rouge_scores_values)
            logger.info(' bleu-4: %f', bleu4)
            logger.info(' rouge : %s', str(rouge_scores_values))

            VAE.train()

    print("Measuring Input distribution...")
    plot_input_distribution(test_loader, num_iters)
    print("Val Setup...")
    val_step(val_loader)
    print("Test: Generate...")
    generate(test_loader, num_iters)
    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + '.pt'))

    while num_iters < args.iterations:
        # Run epoch
        st = time.time()

        # Training
        print('Training loop. Batches:', len(train_loader))
        logger.info('\n----------------------------------------------------------------------')
        logger.info("Training loop.       Batches: %d" % len(train_loader))

        # train_iter = iter(train_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(train_iter)
        train_iter = iter(train_loader)
        with tqdm(total=len(train_loader)) as pbar:
            for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(train_loader):
                if num_iters % 100 == 0:
                    print("CURRENT ITERATION: ", num_iters)

                # NOTE: Swaps all the variables for the bidirectional running of the program
                # if num_iters % args.cycle >= args.cycle - args.beta_warmup:
                #     beta = min(1.0, beta + (1. - args.beta_0) / args.beta_warmup)

                if not tuning_all and num_iters >= tuning_all_after_iters:
                    for name, parameter in VAE.named_parameters():
                        # print((name, parameter.requires_grad))
                        parameter.requires_grad = True
                    tuning_all = True

                # This computes a training step going from input to output and computes the losses
                # NORMAL LOSS, Prompt -> Story
                output_forward = train_step(VAE, optimizer, x_mask, x_tokens, y_mask, y_tokens,
                        input_tokens, target_tokens, mask, loss_fn, beta, args.model_type)
                loss_forward, ce_loss_forward, kl_loss_forward = output_forward[-1]

                # BIDIRECTIONAL LOSSES

                # This finds the total loss for the previous sentence, Sentence B -> Sentence A and Sentence A -> Sentence B
                previous_sentence_loss_output = bidirectional_loss("previous_sentence", VAE, optimizer, x_mask,
                    x_tokens, mask, loss_fn, beta, args.model_type, tokenizer, curr_batch_size, curr_seq_len)
                (total_loss_sentence_b_a, total_loss_sentence_a_b, total_ce_loss_sentence_b_a,
                total_ce_loss_sentence_a_b, total_kl_loss_sentence_b_a, total_kl_loss_sentence_a_b) = previous_sentence_loss_output

                # This finds the total loss for all previous sentences, Sentence B -> All Previous Sentences
                all_previous_sentences_loss_output = bidirectional_loss("all_previous_sentences", VAE, optimizer, x_mask,
                    x_tokens, mask, loss_fn, beta, args.model_type, tokenizer, curr_batch_size, curr_seq_len)
                (total_loss_all_previous_sentences, total_ce_loss_all_previous_sentences, total_kl_loss_sentence_all_previous_sentences) = all_previous_sentences_loss_output

                # PROMPT LEVEL LOSS, Story -> Prompt
                output_prompt_backward = train_step(VAE, optimizer, y_mask, y_tokens, x_mask, x_tokens,
                    target_tokens, input_tokens, mask, loss_fn, beta, args.model_type)

                loss_prompt_backward, ce_loss_prompt_backward, kl_loss_prompt_backward = output_prompt_backward[-1]

                # This finds the total loss for the next sentence, Sentence A -> Sentence B and Sentence B -> Sentence A
                # And the total loss for all next sentences, Sentence A -> All Next Sentences
                # And the total loss for all previous sentences, Sentence B -> All Previous Sentences
                loss = loss_forward + total_loss_sentence_b_a + total_loss_sentence_a_b + loss_prompt_backward
                ce_loss = ce_loss_forward + total_ce_loss_sentence_b_a + total_ce_loss_sentence_a_b + ce_loss_prompt_backward
                kl_loss = kl_loss_forward + total_kl_loss_sentence_b_a + total_kl_loss_sentence_a_b + kl_loss_prompt_backward

                lr = scheduler.get_last_lr()[0]
                # Log to Tensorboard
                t_writer.add_scalar('loss', loss, num_iters)
                t_writer.add_scalar('ppl', math.exp(min(ce_loss, 10)), num_iters)
                t_writer.add_scalar('lr', lr, num_iters)
                t_writer.add_scalar('iter_time', time.time() - st, num_iters)
                t_writer.add_scalar('kl', kl_loss, num_iters)
                t_writer.add_scalar('beta', beta, num_iters)

                if args.model_type == 'ae_vae_fusion':
                    # Output is never defined.  Raise error
                    raise NotImplementedError()
                    loss, ce_loss, kl_loss = output[0]
                    # Log to Tensorboard
                    t_writer.add_scalar('ae_loss', loss, num_iters)
                    t_writer.add_scalar('ae_kl', kl_loss, num_iters)

                st = time.time()
                end = num_iters >= args.iterations

                if args.warmup != -1:
                    scheduler.step()

                if end: break
                num_iters += 1
                pbar.update(1)

                if num_iters % args.cycle == 0:
                    beta = args.beta_0
                    logger.info('KL annealing restart')

                if num_iters % 10000 == 0:
                    plot_input_distribution(test_loader, num_iters)
                    val_step(val_loader)
                    generate(test_loader, num_iters)

                if num_iters % 50000 == 0:
                    print('Saving model...')
                    logger.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
                    logger.info("Saving model...")
                    logger.info('\n------------------------------------------------------')
                    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + '.pt'))

                if args.switch_time > 0 and num_iters == int(args.iterations * args.switch_time):
                    print('Switch to long sequence training')
                    logger.info("Switch to long sequence training")
                    curr_seq_len = args.long_seq_len
                    curr_batch_size = args.train_batch_size
                    train_loader, val_loader, test_loader = prepare_dataset(
                        args.data_dir, args.dataset, tokenizer,
                        args.train_batch_size, curr_seq_len,
                        args.val_batch_size, curr_seq_len,
                        args.test_batch_size, curr_seq_len,
                        make_test=True,
                        num_workers=args.workers, data_type=args.data_type
                    )

        if not end:
            e += 1
            logger.info("Training loop. The ith epoch completed: %d" % e)

    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_latest.pt'))
    print('Training complete.')
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
