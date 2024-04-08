import numpy as np
import re
import json
import os
import time
import random
from collections import defaultdict, deque
import datetime
import math
import torch
import torch.distributed as dist
import warnings


# fix the seed for reproducibility
def set_seed(seed):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
        

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return 0


# Preprocess text, 
# truncate words
def pre_caption(caption,max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


def pre_question(question,max_ques_words=50):
    question = re.sub(
        r"([.!\"()*#:;~])",
        '',
        question.lower(),
    ) 
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question


def save_result(result, result_dir, filename, remove_duplicate=''):
    result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename, get_rank()))
    final_result_file = os.path.join(result_dir, '%s.json'%filename)
    
    json.dump(result,open(result_file,'w'))

    dist.barrier()

    if is_main_process():   
        # combine results from all processes
        result = []

        for rank in range(get_world_size()):
            result_file = os.path.join(result_dir, '%s_rank%d.json'%(filename,rank))
            res = json.load(open(result_file,'r'))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []    
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new             
                
        json.dump(result,open(final_result_file,'w'))            
        print('result file saved to %s'%final_result_file)

    return final_result_file

# for down-stream task fine-tune(e.g., cross-modal retrieval) optimizer
# weight_decay=0.05
# max_epoch=6
def cosine_lr_schedule(optimizer, epoch, max_epoch=6, init_lr=1e-5, min_lr=0.):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# for pre-train optimizer
def warmup_lr_schedule(optimizer, step=0, max_step=3000, init_lr=1e-6, max_lr=3e-4):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    

# for pre-train optimizer
# weight_decay=0.05
# max_epoch=20
def step_lr_schedule(optimizer, epoch, init_lr=3e-4, min_lr=1e-6, decay_rate=0.9):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  


# warmup + cosine_schedule
def warmup_cosine_lr_schedule(opt, optimizer, epoch, warmup_epoch=1, cos_epoch=8, 
                              init_lr=3e-4, warmup_lr=1e-5, min_lr=1e-6):  
    
    if epoch < warmup_epoch:
        lr = warmup_lr
    else:
        lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch-warmup_epoch) / cos_epoch)) + min_lr
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Mask print on non main processes
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
    warnings.filterwarnings('ignore')


# Judge whether distributed training can be used and initialized
def is_dist_avail_and_initialized():
  
    if not dist.is_available():
        return False
    
    if not dist.is_initialized():
        return False
    
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    else:
        return dist.get_world_size()


# 0 to world_size - 1
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    else:
        return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:

        args.rank = int(os.environ["RANK"])
        
        args.world_size = int(os.environ['WORLD_SIZE'])

        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.gpu = args.local_rank

    else:
        print('Not using distributed mode')
        args.distributed = False
        return 0

    args.distributed = True

    torch.cuda.set_device(args.gpu)

    print('distributed init: gpu-id={}, local_rank={}, rank={}, world_size={}, init_method={}'.format(
        args.gpu, args.local_rank, args.rank, args.world_size, args.dist_url), flush=True)
    
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    
    torch.distributed.barrier()

    setup_for_distributed(args.rank == 0)     


def concat_all_gather(tensor, keep_grad=True):

    # There is no need for gather in the single-proc case
    if get_world_size() == 1:
        return tensor

    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    if keep_grad:
        tensors_gather[dist.get_rank()] = tensor

    output = torch.cat(tensors_gather, dim=0)
    
    return output  


# come from BLIP, https://github.com/salesforce/BLIP/blob/b7bb1eeb6e901044a9eb1016f408ee908b216bc7/models/blip_retrieval.py#L306
# Gather tensors from all workers with support for backward propagation:
# This implementation does not cut the gradients as torch.distributed.all_gather does.
class GatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        
        # op=torch.distributed.ReduceOp.SUM
        torch.distributed.all_reduce(all_gradients)

        return all_gradients[torch.distributed.get_rank()]


# Performs all_gather operation on the provided tensors.
# Graph remains connected for backward grad computation.
def all_gather_with_grad(tensors):
    
    # There is no need for reduction in the single-proc case
    if get_world_size() == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


# come from
# https://github.com/Lightning-Universe/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py#L224
# the call method is: 
# features_gather = SyncFunction.apply(features) 
class SyncFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size

        return grad_input[idx_from:idx_to]
    

# come from
# https://github.com/openai/CLIP/issues/111#issuecomment-931955836
# return AllGatherFunction.apply(tensor)
class AllGatherFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, reduce_dtype: torch.dtype = torch.float32):
        ctx.reduce_dtype = reduce_dtype

        output = list(torch.empty_like(tensor) for _ in range(dist.get_world_size()))
        dist.all_gather(output, tensor)
        output = torch.cat(output, dim=0)
        
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        
        grad_dtype = grad_output.dtype
        input_list = list(grad_output.to(ctx.reduce_dtype).chunk(dist.get_world_size()))
        grad_input = torch.empty_like(input_list[dist.get_rank()])
        
        dist.reduce_scatter(grad_input, input_list)
        
        return grad_input.to(grad_dtype)


def gather_result(value):
    if isinstance(value, torch.Tensor):
        torch.distributed.all_reduce(value, async_op=False)  # compute the sum
        value.mul_(1.0 / torch.distributed.get_world_size())  # compute the avg
    return value


if __name__ == '__main__':

    pass