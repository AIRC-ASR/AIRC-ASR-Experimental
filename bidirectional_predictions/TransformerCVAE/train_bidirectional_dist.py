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

import torch.distributed as dist
import importlib
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

nltk.download('punkt')
nltk.download('stopwords')


def main_worker(gpu, ngpus_per_node, args):
    logger = logging.getLogger("transformers")
    if args.model_type == 'cvae':
        args.learn_prior = True
    else:
        args.learn_prior = False

    # GPU
    args.gpu = gpu
    print("There are ", torch.cuda.device_count(), " available GPUs!")
    device = torch.device('cuda', args.gpu)
    torch.cuda.set_device(device)
    Device.set_device(str(args.gpu), args.gpu)
    print('Current single GPU: {}'.format(torch.cuda.current_device()))

    # randomness
    np.random.seed(args.seed)
    prng = np.random.RandomState()
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # For multiprocessing distributed training, rank needs to be the global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu
    print('Setting rank', args.rank)
    recon_attempt = 1
    connected = False
    if args.rank != 0:
        # Stall to have rank 0 node go first
        time.sleep(3)
    
    while not connected:
        try:
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
            connected = True
            print('Established connection. Rank:', args.rank)
        except Exception as e:
            # Sometimes the head node launches after the worker, which would cause an issue
            print('Failed to init process group. Retrying...', recon_attempt, e)
            recon_attempt += 1
            time.sleep(10)

    # logging
    if args.rank == 0:
        save_folder = os.path.join(args.out_dir, args.experiment)
        os.makedirs(save_folder, exist_ok=True)
        t_writer = SummaryWriter(os.path.join(save_folder, 'train'), flush_secs=5)
        v_writer = SummaryWriter(os.path.join(save_folder, 'val'), flush_secs=5)
        importlib.reload(logging)
        logging.basicConfig(filename=os.path.join(save_folder, 'train.log'),
                            level=logging.INFO, format='%(asctime)s--- %(message)s')
        logging.info('\n*******************************************************************************\n')
        logging.info("the configuration:")
        logging.info(str(args).replace(',', '\n'))

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
    special_tokens = {
        'sentence_fwd': '</SFWD/>',
        'sentence_bkwd': '</SBKWD/>'
    }
    # special_tokens_dict = {
    #     'pad_token': '<|startoftext|>',
    #     'cls_token': '<|startofcond|>',
    #     'sep_token': '<|sepofcond|>',
    #     'mask_token': '<|endofcond|>'
    # }
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f'We have added {len(special_tokens.keys())} special tokens')
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
    args.load = args.reload_path
    if args.load:
        logger.info('Loading model weights...')
        state = torch.load(os.path.join(args.load), map_location="cpu")
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
        num_workers=args.workers, data_type=args.data_type,
        distributed=True
    )
    logger.info('Done.')

    logger.info('Wrapping models and optimizers...')

    # Apply linear scaling rule to increase batch size for short sequence training.
    curr_batch_size = args.train_batch_size
    curr_seq_len = args.short_seq_len
    lr_schedule = switch_schedule(linear_schedule(args), curr_batch_size / curr_seq_len,
                                int(args.iterations * args.switch_time))
    VAE = VAE.to(device)
    VAE.train()

    optimizer = torch.optim.AdamW(VAE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    loss_model = DDP(VAE)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    logger.info('Done.')

    logger.info("Begin training iterations")
    max_val_batches = 20000  # max num. of val batches
    logger.info("Total iteration: %d" % args.iterations)
    e = 0  # number of epoch
    num_iters = 0
    i = 0

    if args.load:
        # Resume training from a checkpoint
        train_iter = iter(train_loader)

        e = args.reload_epoch
        num_iters = args.reload_iters
        i = args.reload_batches

        # Fast forward to where we left off in the dataloader
        for _ in range(args.reload_batches):
            next(train_iter)

    logger.info(f"Resume training from epoch {args.reload_epoch}, batch {args.reload_batches}")

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

    def calculate_loss(model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask):
        '''Calculates the loss of the model forward, backward, and for the sentence combinations'''

        # This computes a training step going from input to output and computes the losses
        # NORMAL LOSS, Prompt -> Story
        if args.fwd_loss_weight > 0:
            loss_forward, ce_loss_forward, kl_loss_forward = train_step(model, optimizer, x_mask, x_tokens, y_mask, y_tokens,
                input_tokens, target_tokens, mask, loss_fn, beta, args.model_type)[-1]
        else:
            loss_forward, ce_loss_forward, kl_loss_forward = 0, 0, 0

        # PROMPT LEVEL LOSS, Story -> Prompt
        if args.prompt_loss_weight > 0:
            loss_prompt_backward, ce_loss_prompt_backward, kl_loss_prompt_backward = train_step(model, optimizer, y_mask, y_tokens, x_mask, x_tokens,
                target_tokens, input_tokens, mask, loss_fn, beta, args.model_type)[-1]
        else:
            loss_prompt_backward, ce_loss_prompt_backward, kl_loss_prompt_backward = 0, 0, 0

        # BIDIRECTIONAL LOSSES

        # This finds the total loss for the previous sentence, Sentence B -> Sentence A and Sentence A -> Sentence B
        if args.bkwd_loss_weight > 0:
            previous_sentence_loss_output = bidirectional_loss("previous_sentence", model, optimizer, y_mask,
                y_tokens, mask, loss_fn, beta, args.model_type, tokenizer, curr_batch_size, curr_seq_len, input_tokens)
            (total_loss_sentence_b_a, total_loss_sentence_a_b, total_ce_loss_sentence_b_a,
            total_ce_loss_sentence_a_b, total_kl_loss_sentence_b_a, total_kl_loss_sentence_a_b) = previous_sentence_loss_output
            print('previous_sentence_loss_output', previous_sentence_loss_output)
        else:
            total_loss_sentence_b_a, total_loss_sentence_a_b, total_ce_loss_sentence_b_a, total_ce_loss_sentence_a_b, total_kl_loss_sentence_b_a, total_kl_loss_sentence_a_b = 0, 0, 0, 0, 0, 0
        
        # This finds the total loss for all previous sentences, Sentence B -> All Previous Sentences
        if args.all_sentence_loss_weight > 0:
            all_previous_sentences_loss_output = bidirectional_loss("all_previous_sentences", model, optimizer, y_mask,
                y_tokens, mask, loss_fn, beta, args.model_type, tokenizer, curr_batch_size, curr_seq_len, input_tokens)
            (total_loss_all_previous_sentences, total_ce_loss_all_previous_sentences, total_kl_loss_all_previous_sentences) = all_previous_sentences_loss_output
            print('all_previous_sentences_loss_output', all_previous_sentences_loss_output)
        else:
            total_loss_all_previous_sentences, total_ce_loss_all_previous_sentences, total_kl_loss_all_previous_sentences = 0, 0, 0

        # TOTAL LOSSES
        loss = (args.fwd_loss_weight*loss_forward) + (args.prompt_loss_weight*loss_prompt_backward) + \
            (args.bkwd_loss_weight*total_loss_sentence_b_a) + \
            (args.bkwd_loss_weight*total_loss_sentence_a_b) + (args.all_sentence_loss_weight*total_loss_all_previous_sentences)

        ce_loss = (args.fwd_loss_weight*ce_loss_forward) + (args.prompt_loss_weight*ce_loss_prompt_backward) + \
            (args.bkwd_loss_weight*total_ce_loss_sentence_b_a) + \
            (args.bkwd_loss_weight*total_ce_loss_sentence_a_b) + (args.all_sentence_loss_weight*total_ce_loss_all_previous_sentences)

        kl_loss = (args.fwd_loss_weight*kl_loss_forward) + (args.prompt_loss_weight*kl_loss_prompt_backward) + \
            (args.bkwd_loss_weight*total_kl_loss_sentence_b_a) + \
            (args.bkwd_loss_weight*total_kl_loss_sentence_a_b) + (args.all_sentence_loss_weight*total_kl_loss_all_previous_sentences)

        return loss, ce_loss, kl_loss

    if args.rank == 0:
        # eval_step()
        torch.save(VAE.state_dict(), os.path.join(save_folder,
            f'model_{e}_{num_iters:07d}_{i}_bidirectional_{args.fwd_loss_weight}_{args.bkwd_loss_weight}_{args.all_sentence_loss_weight}_{args.prompt_loss_weight}.pt')
        )

    while e < args.num_epochs:
        while num_iters < args.iterations:
            # Run epoch
            st = time.time()

            # Training
            logger.info('\n----------------------------------------------------------------------')
            logger.info("Training loop.       Batches: %d" % len(train_loader))

            with tqdm(total=len(train_loader)) as pbar:
                while i < len(train_loader):
                    (x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask) = next(train_iter)
                    # NOTE: Swaps all the variables for the bidirectional running of the program
                    # if num_iters % args.cycle >= args.cycle - args.beta_warmup:
                    #     beta = min(1.0, beta + (1. - args.beta_0) / args.beta_warmup)

                    if not tuning_all and num_iters >= tuning_all_after_iters:
                        for name, parameter in VAE.named_parameters():
                            # logger.info((name, parameter.requires_grad))
                            parameter.requires_grad = True
                        tuning_all = True

                    try:
                        loss, ce_loss, kl_loss = calculate_loss(loss_model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask)
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            logger.info('| WARNING: ran out of memory, skipping batch')
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            raise e

                    if args.rank == 0 and num_iters % 100 == 0:
                        logger.info(f"CURRENT ITERATION: {num_iters}")
                        logger.info(f"CURRENT LOSS: Loss: {loss}, CE: {ce_loss}, KL: {kl_loss}")

                    if args.rank == 0:
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
                    
                    num_iters += 1
                    pbar.update(1)

                    if num_iters % args.cycle == 0:
                        beta = args.beta_0
                        logger.info('KL annealing restart')

                    if args.rank == 0 and num_iters % 10000 == 0:
                        eval_step()

                    if args.rank == 0 and num_iters % 10000 == 0:
                        logger.info('Saving model...')
                        logger.info("Iteration completed: %d, remained %d" % (num_iters, args.iterations - num_iters))
                        logger.info("Saving model...")
                        logger.info('\n------------------------------------------------------')
                        torch.save(VAE.state_dict(), os.path.join(save_folder,
                            f'model_{e}_{num_iters:07d}_{i}_bidirectional_{args.fwd_loss_weight}_{args.bkwd_loss_weight}_{args.all_sentence_loss_weight}_{args.prompt_loss_weight}.pt')
                        )

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
                            num_workers=args.workers, data_type=args.data_type,
                            distributed=True
                        )

                    i += 1

        e += 1
        logger.info("Training loop. The ith epoch completed: %d" % e)

    if args.rank == 0:
        torch.save(VAE.state_dict(), os.path.join(save_folder,
            f'model_{e}_{num_iters:07d}_{i}_bidirectional_{args.fwd_loss_weight}_{args.bkwd_loss_weight}_{args.all_sentence_loss_weight}_{args.prompt_loss_weight}.pt')
        )
    logger.info("Training complete.")
    dist.destroy_process_group()

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

    # Loss weighting args
    parser.add_argument('--fwd_loss_weight', type=float, default=1, help="Weight multiplier for forward loss.")
    parser.add_argument('--bkwd_loss_weight', type=float, default=1, help="Weight multiplier for backward loss.")
    parser.add_argument('--all_sentence_loss_weight', type=float, default=1, help="Weight multiplier for all previous sentence loss (0 to A -> B).")
    parser.add_argument('--prompt_loss_weight', type=float, default=1, help="Weight multiplier for backward prompt loss.")
    
    # Reload args
    parser.add_argument('--reload_path', type=str, default='')
    parser.add_argument('--reload_epoch', type=int, default=0)
    parser.add_argument('--reload_iters', type=int, default=0)
    parser.add_argument('--reload_batches', type=int, default=0)

    parser.add_argument('--num_epochs', type=int, default=4)

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:9999', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')


    # NOTE: Use for changing the arguments of the program
    '''
    1 0 0 0
    1 2 0 0
    1 0 2 0
    1 0 0 2
    '''
    # args = parser.parse_args('test --add_input --learn_prior --fp16 --switch-time 0.5 '
    #                          '--val_batch_size 1 --test_batch_size 1 '
    #                          '--short_seq_len 1024 --long_seq_len 1024 '
    #                          '--fwd_loss_weight 1 --bkwd_loss_weight 0 --all_sentence_loss_weight 2 '
    #                          '--prompt_loss_weight 0 '.split())
    args = parser.parse_args()



    # Each node is expected to have same number of GPUs    
    ngpus_per_node = torch.cuda.device_count()

    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size

    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

if __name__ == "__main__":
    main()
