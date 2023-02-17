import torch
import torch.nn.functional as F
from torch.cuda import amp
from data.util import *
from util import *
from model import *
from collections import Counter
import matplotlib
matplotlib.use('Agg')


class Device:
    ident: str
    type: str
    device: torch.device

    @classmethod
    def set_device(self, ident, type):
        self.device_id = ident
        self.type = type
        os.environ["CUDA_VISIBLE_DEVICES"] = ident
        self.device = torch.device(type)


def compute_loss(model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(Device.device)
    target_tokens = target_tokens.to(Device.device)
    mask = mask.to(Device.device)
    x_mask = x_mask.to(Device.device)
    x_tokens = x_tokens.to(Device.device)
    y_mask = y_mask.to(Device.device)
    y_tokens = y_tokens.to(Device.device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, x_mask=x_mask, x_tokens=x_tokens, y_mask=y_mask,
                    y_tokens=y_tokens)
    logits = outputs[0]
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(Device.device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean() + beta * kl_loss

    return loss, ce_loss, kl_loss


def compute_loss_ae(model, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta):
    input_tokens = input_tokens.to(Device.device)
    target_tokens = target_tokens.to(Device.device)
    mask = mask.to(Device.device)
    x_mask = x_mask.to(Device.device)
    x_tokens = x_tokens.to(Device.device)

    outputs = model(input_ids=input_tokens, attention_mask=mask, y_mask=x_mask, y_tokens=x_tokens, from_mean=True, from_prior=False)

    logits = outputs[0]
    kl_loss = outputs[-1]
    num_logits = logits.size(-1)

    # Perform masking
    if mask is not None:
        mask = mask.type(torch.bool)
        mask = mask.to(Device.device)
        logits = logits.masked_select(mask.unsqueeze(-1))
        target_tokens = target_tokens.masked_select(mask)

    ce_loss = loss_fn(logits.view(-1, num_logits), target_tokens.view(-1))
    kl_loss = kl_loss.mean()
    loss = ce_loss.mean()

    return loss, ce_loss, kl_loss


def train_step(model, optimizer, x_mask, x_tokens, y_mask, y_tokens, input_tokens, target_tokens, mask, loss_fn, beta, model_type):
    output = []
    scaler = amp.GradScaler()
    if model_type == 'ae_vae_fusion':
        optimizer.zero_grad()
        loss, ce_loss, kl_loss = compute_loss_ae(model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
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
    loss, ce_loss, kl_loss = compute_loss(model, x_mask, x_tokens, y_mask, y_tokens, input_tokens,
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


@DeprecationWarning
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