import os, time, gc, json, pickle, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn import DataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup, Conv1D
from tensorboardX import SummaryWriter
from tqdm import tqdm
import importlib
# import logging
from transformers.utils import logging
logger = logging.get_logger("transformers")
import copy

# from apex.optimizers import FusedAdam
# from apex import amp
from torch.cuda import amp
# from apex.fp16_utils import FP16_Optimizer

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
nltk.download('punkt')
nltk.download('stopwords')

devices = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = devices


def compute_loss(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, x_mask=x_mask, x_tokens=x_tokens, y_mask=y_mask,
                    y_tokens=y_tokens)
    logits = outputs[0]
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean() + beta * kl_loss

    return loss, ce_loss, kl_loss


def compute_loss_ae(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(device)
    target_tokens = target_tokens.to(device)
    mask = mask.to(device)
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, y_mask=x_mask, y_tokens=x_tokens, from_mean=True, from_prior=False)

    logits = outputs[0]
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean()

    return loss, ce_loss, kl_loss


def train_step(device, model, optimizer, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta, model_type):
    output = []
    scaler = amp.GradScaler()
    if model_type == 'ae_vae_fusion':
        optimizer.zero_grad()
        loss, ce_loss, kl_loss = compute_loss_ae(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                              target_tokens, mask, loss_fn, beta)
        scaler.scale(loss).backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)  # max_grad_norm=1.0
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        output.append((loss.item(), ce_loss.mean().item(), kl_loss.item()))

    optimizer.zero_grad()
    loss, ce_loss, kl_loss = compute_loss(device, model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
                                          target_tokens, mask, loss_fn, beta)
    # with amp.scale_loss(loss, optimizer) as scaled_loss:
    #     scaled_loss.backward()
    scaler.scale(loss).backward()
    # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 5.0)  # max_grad_norm=1.0
    # loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # max_grad_norm=1.0
    # optimizer.step()
    scaler.step(optimizer)
    scaler.update()
    output.append((loss.item(), ce_loss.mean().item(), kl_loss.item()))

    return output


def top_k_top_p_filtering(logits, top_k=100, top_p=0.95, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def repeat_score(text, ngram=[3, 4, 5, 6]):
    ngram_list = []
    for ng in ngram:
        ngram_list.append([text[idx:idx + ng] for idx in range(len(text) - ng - 1)])

    max_occurs = []
    for ngrams in ngram_list:
        count_result = Counter([' '.join(n) for n in ngrams])
        try:
            max_occurs.append(
                max(count_result.values())
            )
        except:
            pass

    scores = [max_oc / ((len(text) / ngram[idx]) + ngram[idx]) for idx, max_oc in enumerate(max_occurs)]
    return max(scores) if len(scores) >= 1 else 1.0


def sample_sequence(model, tokenizer, length, batch_size=None, x_mask=None, x_tokens=None, y_mask=None, y_tokens=None,
                    temperature=1, top_k=100, top_p=0.95, device='cuda', sample=True, eos_token=None, model_type='cvae'):
    x_mask = x_mask.to(device)
    x_tokens = x_tokens.to(device)
    y_mask = y_mask.to(device)
    y_tokens = y_tokens.to(device)

    with torch.no_grad():
        if model_type == 'cvae':
            try:
                prior_mean, prior_logvar = model.encoder_prior(input_ids=x_tokens, attention_mask=x_mask)[:2]
            except:
                prior_mean = prior_logvar = torch.zeros([batch_size, model.config.n_embd], device=device)
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
        probability = torch.tensor([], dtype=z.dtype, device=device)
        if_end = torch.tensor([False] * batch_size, dtype=torch.bool, device=device)

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

    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[1],
                        help='batch size per GPU. Lists the schedule.')
    parser.add_argument('--seq-lens', nargs='+', type=int, default=[1024],
                        help='seq length per sample. Lists the schedule.')
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

    # NOTE: Use for changing the arguments of the program
    args = parser.parse_args('test --batch-sizes 1 1 --seq-lens 1024 1024 '
                             '--add_input --learn_prior --fp16 --iterations 10 --switch-time 0.1'.split()) # wi.12.proj_vary_beta_cvae

    if args.model_type == 'cvae':
        args.learn_prior = True
    else:
        args.learn_prior = False

    # GPU
    if not torch.cuda.is_available(): args.no_gpu = True
    gpu = not args.no_gpu
    if gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        # print('Setting GPUs {}'.format(args.device))
        print('Using GPU devices {}'.format(devices))
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    device = torch.device(args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
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
    # Batch and sequence length schedule
    assert len(args.batch_sizes) == len(args.seq_lens)
    batch_schedule = list(zip(map(int, args.batch_sizes), map(int, args.seq_lens)))
    assert len(batch_schedule) <= 2, 'Currently not supporting multiple schedule'
    cur_b_schedule = len(batch_schedule) - 1 if args.switch_time == 0 else 0
    print('Batch schedule', batch_schedule, cur_b_schedule, args.seq_lens)
    train_loader, val_loader, test_loader = prepare_dataset(
        args.data_dir, args.dataset, tokenizer,
        batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
        batch_schedule[-1][0], batch_schedule[-1][1],
        batch_schedule[-1][0], batch_schedule[-1][1],
        make_test=True,
        num_workers=args.workers, data_type=args.data_type
    )
    print('Done.')

    ###
    val_loader = test_loader
    ###

    print('Wrapping models and optimizers...')

    # Apply linear scaling rule to increase batch size for short sequence training.
    lr_schedule = switch_schedule(linear_schedule(args), batch_schedule[cur_b_schedule][0] / batch_schedule[-1][0],
                                  int(args.iterations * args.switch_time))
    VAE = VAE.to(device)
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
                        loss, ce_loss, kl_loss = compute_loss(device, VAE, x_mask, x_tokens, y_mask, y_tokens,
                                                              input_tokens, target_tokens, mask, loss_fn, 1.0)
                    else:
                        loss, ce_loss, kl_loss = compute_loss_ae(device, VAE, x_mask, x_tokens, y_mask, y_tokens,
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

        VAE.train()

    def test_plot(test_loader, num_iters):
        VAE.eval()

        # get embedding
        X_emb = None
        y = None

        # test_iter = iter(test_loader); x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask = next(test_iter)
        with tqdm(total=len(test_loader)) as pbar:
            for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(
                    test_loader):
                y_mask = y_mask.to(device)
                y_tokens = y_tokens.to(device)
                x_mask = x_mask.to(device)
                x_tokens = x_tokens.to(device)
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
                        tokenizer=tokenizer,
                        length=length,
                        batch_size=args.batch_size,
                        x_mask=x_mask,
                        x_tokens=x_tokens,
                        y_mask=y_mask,
                        y_tokens=y_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        device=device,
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

    test_plot(test_loader, num_iters)
    val_step(val_loader)
    generate(test_loader, num_iters)
    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + '.pt'))

    def find_loss_bidirectional(
        loss_type, device, VAE, optimizer, y_mask, y_tokens, x_mask, x_tokens,
        target_tokens, input_tokens, mask, loss_fn, beta, model_type
    ):
        '''This function finds the bidirectional loss on different levels.
        loss_types designates the possible loss types: 
            "previous_sentence": The latest sentence needs to predict the previous one and vice versa.
            "previous_sentences" The latest sentence needs to predict the previous ones and vice versa.
            "prompt": The prompt predicts the target story and vice versa.
        All other arguments are the same as train_step()
        '''
        if loss_type == "previous_sentence":
            pass
        elif loss_type == "previous_sentences":
            pass
        elif loss_type == "prompt":
            return train_step(device, VAE, optimizer, y_mask, y_tokens, x_mask, x_tokens,
                target_tokens, input_tokens, mask, loss_fn, beta, model_type)
        
        return None

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
                print("CURRENT ITERATION: ", num_iters)

                y_tokens_text = tokenizer.decode(y_tokens[0].tolist())
                y_sentences = y_tokens_text.split('.')

                y_mask_text = tokenizer.decode(y_mask[0].tolist())

                split_indices = []
                print('y_tokens shape', y_tokens.shape)
                print('y_mask shape', y_mask.shape)
                torch.set_printoptions(threshold=10000)

                print('y_mask', y_mask[0])
                for y_sentence in y_sentences:
                    # print('y_sentence',  y_tokens_text[total_len:total_len + len(y_sentence) + 1])
                    y_sentence_encoded = tokenizer.encode(y_sentence + '.')
                    print('y_sentence_encoded', y_sentence_encoded)
                    # y_sentence_decoded = tokenizer.decode(y_sentence_encoded)
                    # print('y_sentence_decoded', y_sentence_decoded)
                    y_mask_a = torch.ones(len(y_sentence_encoded), dtype=torch.long).to(device)
                    print('y_mask_a', y_mask_a)
                    print('LEN', len(y_sentence_encoded), len(y_mask_a))
                    assert(len(y_sentence_encoded) == len(y_mask_a))
                break
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
                output_forward = train_step(device, VAE, optimizer, x_mask_a, x_tokens_a, y_mask_a, y_tokens_a,
                        input_tokens_a, target_tokens_a, mask_a, loss_fn, beta, args.model_type)
                loss_forward, ce_loss_forward, kl_loss_forward = output_forward[-1]

                # # SENTENCE LEVEL LOSS, Sentence A -> Sentence B
                # output_sentence_a_b = train_step(device, VAE, optimizer, x_mask_a, x_tokens_a, y_mask_b, y_tokens_b,
                #         input_tokens_a, target_tokens_b, mask_a, loss_fn, beta, args.model_type)
                # loss_sentence_a_b, ce_loss_sentence_a_b, kl_loss_sentence_a_b = output_sentence_a_b[-1]

                # # SENTENCE LEVEL LOSS, Sentence B -> Sentence A
                # output_sentence_b_a = train_step(device, VAE, optimizer, x_mask_b, x_tokens_b, y_mask_a, y_tokens_a,
                #         input_tokens_b, target_tokens_a, mask_b, loss_fn, beta, args.model_type)
                # loss_sentence_b_a, ce_loss_sentence_b_a, kl_loss_sentence_b_a = output_sentence_b_a[-1]

                # # SENTENCE TO PROMPT LEVEL LOSS, Sentence A -> Prompt A
                # output_sentence_a_prompt_a = train_step(device, VAE, optimizer, y_mask_a, y_tokens_a, x_mask_a, x_tokens_a,
                #         target_tokens_a, input_tokens_a, mask_a, loss_fn, beta, args.model_type)
                # loss_sentence_a_prompt_a, ce_loss_sentence_a_prompt_a, kl_loss_sentence_a_prompt_a = output_sentence_a_prompt_a[-1]

                # # This finds the overall loss by summing over the forward and sentence level losses
                # loss = loss_forward + loss_sentence_a_b + loss_sentence_b_a
                # ce_loss = ce_loss_forward + ce_loss_sentence_a_b + ce_loss_sentence_b_a
                # kl_loss = kl_loss_forward + kl_loss_sentence_a_b + kl_loss_sentence_b_a

                loss = loss_forward
                ce_loss = ce_loss_forward
                kl_loss = kl_loss_forward


                lr = scheduler.get_last_lr()[0]
                # Log to Tensorboard
                t_writer.add_scalar('loss', loss, num_iters)
                t_writer.add_scalar('ppl', math.exp(min(ce_loss, 10)), num_iters)
                t_writer.add_scalar('lr', lr, num_iters)
                t_writer.add_scalar('iter_time', time.time() - st, num_iters)
                t_writer.add_scalar('kl', kl_loss, num_iters)
                t_writer.add_scalar('beta', beta, num_iters)

                if args.model_type == 'ae_vae_fusion':
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
                    test_plot(test_loader, num_iters)
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
                    # TODO: Figure out why this causes an index error
                    cur_b_schedule += 1
                    train_loader, val_loader, test_loader = prepare_dataset(
                        args.data_dir, args.dataset, tokenizer,
                        batch_schedule[cur_b_schedule][0], batch_schedule[cur_b_schedule][1],
                        batch_schedule[-1][0], batch_schedule[-1][1],
                        batch_schedule[-1][0], batch_schedule[-1][1],
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
