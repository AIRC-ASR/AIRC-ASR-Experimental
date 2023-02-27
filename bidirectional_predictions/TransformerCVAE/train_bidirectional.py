import logging
import os, time, gc, argparse, math
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, Conv1D
from tensorboardX import SummaryWriter
from tqdm import tqdm
import copy
from util import init_para_frompretrained, num_params, prepare_dataset, linear_schedule, switch_schedule
from model import VAEModel
import nltk
from bi_training_core import train_step, Device
from bi_loss import bidirectional_loss
from bi_eval_step import validate_step, plot_input_distribution, generate_samples

nltk.download('punkt')
nltk.download('stopwords')
# devices = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = devices


def main():
    logger = logging.getLogger("transformers")

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

    # Loss weighting args
    parser.add_argument('--fwd_loss_weight', type=float, default=1, help="Weight multiplier for forward loss.")
    parser.add_argument('--bkwd_loss_weight', type=float, default=0, help="Weight multiplier for backward loss.")
    parser.add_argument('--fwd_sentence_loss_weight', type=float, default=0, help="Weight multiplier for forward sentence loss (A -> B).")
    parser.add_argument('--bkwd_sentence_loss_weight', type=float, default=0, help="Weight multiplier for forward backward loss (B -> A).")
    parser.add_argument('--all_sentence_loss_weight', type=float, default=0, help="Weight multiplier for all previous sentence loss (0 to A -> B).")

    # NOTE: Use for changing the arguments of the program
    args = parser.parse_args('test --add_input --learn_prior --fp16 --iterations 20000 --switch-time 0.5 '
                             '--train_batch_size 1 --val_batch_size 1 --test_batch_size 1 '
                             '--short_seq_len 1024 --long_seq_len 1024 '
                             '--fwd_loss_weight 1.0 --bkwd_sentence_loss_weight 0 --all_sentence_loss_weight 0'.split())

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
        logger.info(f"There are {torch.cuda.device_count()} available GPUs!")
        logger.info('Using GPU devices {}'.format(devices))
        torch.cuda.set_device(args.gpu)
        logger.info('Current single GPU: {}'.format(torch.cuda.current_device()))

    Device.set_device(devices, args.gpu if gpu else "cpu")

    # randomness
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if gpu: torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    logger.info('\n*******************************************************************************\n')
    logger.debug("the configuration:")
    logger.debug(str(args).replace(',', '\n'))

    logger.info('Loading models...')

    logger.setLevel(logging.WARNING)
    save_folder = os.path.join(args.out_dir, args.experiment)
    os.makedirs(save_folder, exist_ok=True)
    t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
    # importlib.reload(logger)
    # logger.basicConfig(filename=os.path.join(save_folder, 'train.log'), level=logger.INFO, format='%(asctime)s--- %(message)s')
    cache_dir = os.path.join(args.out_dir, 'model_cache')
    os.makedirs(cache_dir, exist_ok=True)
    # Load pre-trained teacher tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
    # Hack to allow tokenizing longer sequences.
    tokenizer.max_len = int(1e12)
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir=cache_dir)
    logger.info(f'gpt2_params: {num_params(gpt2_model)}')  # gpt2: 124439808
    config = GPT2Config()
    config.n_ctx = 1024

    # add special tokens
    special_tokens = {
        'sentence_fwd': '<SFWD>',
        'sentence_bkwd': '<SBKWD>'
    }
    # special_tokens_dict = {
    #     'pad_token': '<|startoftext|>',
    #     'cls_token': '<|startofcond|>',
    #     'sep_token': '<|sepofcond|>',
    #     'mask_token': '<|endofcond|>'
    # }
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info('We have added', len(special_tokens), 'special tokens')
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
    logger.setLevel(logging.INFO)
    logger.info(f'VAE_params: {num_params(VAE)}')  # 286694400
    if args.load:
        logger.info('Loading model weights...')
        state = torch.load(os.path.join(args.load, 'model_latest.pt'), map_location=args.gpu if gpu else "cpu") # model_latest.pt
        if 'module' in list(state.keys())[0]:  # model_path is data parallel model with attr 'module'
            state_copy = copy.copy(state)
            keys = state_copy.keys()
            for k in keys:
                state[k.replace('module.', '')] = state.pop(k)
        VAE.load_state_dict(state)
        gc.collect()
    logger.info('Done.')

    # fix pre-trained parameters before certain iterations
    tuning_all_after_iters = 40000
    tuning_all = False
    for name, parameter in VAE.named_parameters():
        # logger.info((name, parameter.requires_grad))
        new_pars = ['c_z', 'attention_weights', 'mean', 'logvar', 'input_proj', 'attn_proj', 'Nu_fc1', 'Nu_fc2', 'lm_head_rep']

        if not any([True if n in name else False for n in new_pars]):
           parameter.requires_grad = False

    logger.info('Setup data...')
    curr_seq_len = args.short_seq_len
    train_loader, val_loader, test_loader = prepare_dataset(
        args.data_dir, args.dataset, tokenizer,
        args.train_batch_size, curr_seq_len,
        args.val_batch_size, curr_seq_len,
        args.test_batch_size, curr_seq_len,
        make_test=True,
        num_workers=args.workers, data_type=args.data_type
    )
    logger.info('Done.')

    logger.info('Wrapping models and optimizers...')

    # Apply linear scaling rule to increase batch size for short sequence training.
    curr_batch_size = args.train_batch_size
    curr_seq_len = args.short_seq_len
    lr_schedule = switch_schedule(linear_schedule(args), curr_batch_size / curr_seq_len,
                                  int(args.iterations * args.switch_time))
    VAE = VAE.to(Device.device)
    VAE.train()

    optimizer = torch.optim.AdamW(VAE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    logger.info('Done.')

    logger.info("Begin training iterations")
    max_val_batches = 20000  # max num. of val batches
    logger.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    optimizer.zero_grad()
    beta = args.beta_0

    def eval_step():
        '''Evaluates the performance of the model after a training step'''

        logger.info("Measuring Input distribution...")
        plot_input_distribution(VAE, tokenizer, args.model_type, test_loader, args.dataset, num_iters, save_folder)
        logger.info("Validation Step...")
        validate_step(VAE, tokenizer, args.model_type, val_loader, num_iters, max_val_batches, loss_fn, save_folder)
        logger.info("Generate output samples...")
        generate_samples(VAE, tokenizer, args, test_loader, num_iters, save_folder)

    def calculate_loss(x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask):
        '''Calculates the loss of the model forward, backward, and for the sentence combinations'''

        # This computes a training step going from input to output and computes the losses
        # NORMAL LOSS, Prompt -> Story
        loss_forward, ce_loss_forward, kl_loss_forward = 0, 0, 0
        loss_backward, ce_loss_backward, kl_loss_backward = 0, 0, 0
        """loss_forward, ce_loss_forward, kl_loss_forward = train_step(VAE, optimizer, x_mask, x_tokens, y_mask, y_tokens,
            input_tokens, target_tokens, mask, loss_fn, beta, args.model_type)[-1]

        # PROMPT LEVEL LOSS, Story -> Prompt
        loss_backward, ce_loss_backward, kl_loss_backward = train_step(VAE, optimizer, y_mask, y_tokens, x_mask, x_tokens,
            target_tokens, input_tokens, mask, loss_fn, beta, args.model_type)[-1]
"""
        # BIDIRECTIONAL LOSSES

        # This finds the total loss for the previous sentence, Sentence B -> Sentence A and Sentence A -> Sentence B
        previous_sentence_loss_output = bidirectional_loss("previous_sentence", VAE, optimizer, x_mask,
            x_tokens, mask, loss_fn, beta, args.model_type, tokenizer, curr_batch_size, curr_seq_len, input_tokens)
        (total_loss_sentence_b_a, total_loss_sentence_a_b, total_ce_loss_sentence_b_a,
        total_ce_loss_sentence_a_b, total_kl_loss_sentence_b_a, total_kl_loss_sentence_a_b) = previous_sentence_loss_output

        # This finds the total loss for all previous sentences, Sentence B -> All Previous Sentences
        all_previous_sentences_loss_output = bidirectional_loss("all_previous_sentences", VAE, optimizer, x_mask,
            x_tokens, mask, loss_fn, beta, args.model_type, tokenizer, curr_batch_size, curr_seq_len, input_tokens)
        (total_loss_all_previous_sentences, total_ce_loss_all_previous_sentences, 
        total_kl_loss_all_previous_sentences) = all_previous_sentences_loss_output

        # TOTAL LOSSES
        loss = (args.fwd_loss_weight*loss_forward) + (args.bkwd_loss_weight*loss_backward) + \
            (args.bkwd_sentence_loss_weight*total_loss_sentence_b_a) + \
            (args.fwd_sentence_loss_weight*total_loss_sentence_a_b) + (args.all_sentence_loss_weight*total_loss_all_previous_sentences)

        ce_loss = (args.fwd_loss_weight*ce_loss_forward) + (args.bkwd_loss_weight*ce_loss_backward) + \
            (args.bkwd_sentence_loss_weight*total_ce_loss_sentence_b_a) + \
            (args.fwd_sentence_loss_weight*total_ce_loss_sentence_a_b) + (args.all_sentence_loss_weight*total_ce_loss_all_previous_sentences)

        kl_loss = (args.fwd_loss_weight*kl_loss_forward) + (args.bkwd_loss_weight*kl_loss_backward) + \
            (args.bkwd_sentence_loss_weight*total_kl_loss_sentence_b_a) + \
            (args.fwd_sentence_loss_weight*total_kl_loss_sentence_a_b) + (args.all_sentence_loss_weight*total_kl_loss_all_previous_sentences)

        return loss, ce_loss, kl_loss

    # eval_step()
    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + f'_bidirectional_{args.fwd_loss_weight}_{args.bkwd_loss_weight}_{args.all_sentence_loss_weight}' + '.pt'))

    e = 0
    while num_iters < args.iterations:
        # Run epoch
        st = time.time()

        # Training
        logger.info('\n----------------------------------------------------------------------')
        logger.info("Training loop.       Batches: %d" % len(train_loader))

        with tqdm(total=len(train_loader)) as pbar:
            for i, (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) in enumerate(train_loader):
                # NOTE: Swaps all the variables for the bidirectional running of the program
                # if num_iters % args.cycle >= args.cycle - args.beta_warmup:
                #     beta = min(1.0, beta + (1. - args.beta_0) / args.beta_warmup)

                if not tuning_all and num_iters >= tuning_all_after_iters:
                    for name, parameter in VAE.named_parameters():
                        # logger.info((name, parameter.requires_grad))
                        parameter.requires_grad = True
                    tuning_all = True

                try:
                    loss, ce_loss, kl_loss = calculate_loss(x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logger.info('| WARNING: ran out of memory, skipping batch')
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    else:
                        raise e

                if num_iters % 100 == 0:
                    logger.info(f"CURRENT ITERATION: {num_iters}")
                    logger.info(f"CURRENT LOSS: Loss: {loss}, CE: {ce_loss}, KL: {kl_loss}")

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

                if args.warmup != -1:
                    scheduler.step()
                
                end = num_iters >= args.iterations
                if end: break
                num_iters += 1
                pbar.update(1)

                if num_iters % args.cycle == 0:
                    beta = args.beta_0
                    logger.info('KL annealing restart')

                if num_iters % 10000 == 0:
                    eval_step()

                if num_iters % 25000 == 0:
                    logger.info('Saving model...')
                    logger.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
                    logger.info("Saving model...")
                    logger.info('\n------------------------------------------------------')
                    torch.save(VAE.state_dict(), os.path.join(save_folder, 'model_' + '{:07d}'.format(num_iters) + '.pt'))

                if args.switch_time > 0 and num_iters == int(args.iterations * args.switch_time):
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
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
