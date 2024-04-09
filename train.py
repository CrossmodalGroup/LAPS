import os
import time
import numpy as np
import torch
from lib import image_caption, utils
from transformers import BertTokenizer
import logging
import tensorboard_logger as tb_logger
import arguments

from lib import evaluation
from lib.vse import VSEModel, create_optimizer
from lib.evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, shard_attn_scores
from torch.nn.utils import clip_grad_norm_


def main():

    # Hyper Parameters    
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    
    # the path of saving model ckpts and train logs
    opt.model_name = opt.logger_name  

    # Set GPU
    if opt.multi_gpu:
        utils.init_distributed_mode(opt)
        # set seed
        # seed = opt.seed + utils.get_rank()
        # utils.set_seed(seed)
    else:
        torch.cuda.set_device(opt.gpu_id)    

    if utils.is_main_process() and (not os.path.exists(opt.model_name)):
        os.makedirs(opt.model_name)
    
    if utils.is_main_process():
        logging.basicConfig(filename=os.path.join(opt.logger_name, 'train.log'), 
                            filemode='w', format='%(asctime)s %(message)s', level=logging.INFO)
    
    logger = logging.getLogger(__name__)

    if utils.is_main_process():
        logger.info(opt)  
        arguments.save_parameters(opt, opt.logger_name)
        tb_logger.configure(opt.logger_name, flush_secs=5)

    # tokenizer for texts
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
    # opt.vocab_size = len(tokenizer.vocab)
    # vocab_size of BERT model: 30522
    # print('vocab_size of BERT model:', opt.vocab_size)

    # load dataset
    # train-set 
    train_loader = image_caption.get_train_loader(opt, opt.data_path, tokenizer, opt.batch_size, opt.workers, 'train')
    print('Number of images for train-set:', train_loader.dataset.num_images)

    # test-set
    split = 'testall' if opt.dataset == 'coco' else 'test'
    test_loader = image_caption.get_test_loader(opt, opt.data_path, tokenizer, opt.batch_size, opt.workers, split)

    # load model
    model = VSEModel(opt).cuda()

    # get the optimizer
    optimizer = create_optimizer(opt, model)

    start_epoch = 0

    # multi-gpu
    if opt.multi_gpu:
        print('use multi gpu')
        model = torch.nn.parallel.DistributedDataParallel(module=model, 
                                                          device_ids=[opt.gpu], 
                                                          output_device=opt.gpu, 
                                                          find_unused_parameters=True,
                                                          )
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    best_rsum = 0
    
    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):

        if opt.multi_gpu:
            train_loader.sampler.set_epoch(epoch)

        if utils.is_main_process() and epoch == 0:
            logger.info('Log saving path: ' + opt.logger_name)
            logger.info('Models saving path: ' + opt.model_name)

        adjust_learning_rate(opt, optimizer, epoch)

        # # set hard negative for vse loss
        if (epoch >= opt.vse_mean_warmup_epochs) and (opt.loss == 'vse'):
            model_without_ddp.set_max_violation(max_violation=True)

        # train for one epoch
        train(opt, train_loader, model, model_without_ddp, optimizer, epoch)
            
        # evaluate on validation set
        rsum = validate(opt, test_loader, model_without_ddp)

        if utils.is_main_process(): 
            # remember best results and save checkpoint
            is_best = rsum > best_rsum
            best_rsum = max(rsum, best_rsum)

            logger.info("Epoch: [{}], Best rsum: {:.1f} \n".format(epoch, best_rsum))
            state = {'model': model_without_ddp.state_dict(), 'opt': opt, 'epoch': epoch + 1, 
                     'best_rsum': best_rsum, 'Eiters': model_without_ddp.Eiters,
                    }
            save_checkpoint(state, is_best, prefix=opt.model_name)

        # waiting for synchronization
        if opt.multi_gpu:
            torch.distributed.barrier() 
            torch.cuda.empty_cache()   

    # start eval
    if utils.is_main_process() and opt.eval:
        print('Evaluate the model now.')

        base = opt.logger_name
        logging.basicConfig(filename=os.path.join(base, 'eval.log'), filemode='w', 
                            format='%(asctime)s %(message)s', level=logging.INFO, force=True)

        logger = logging.getLogger()
        logger.info('Evaluating {}...'.format(base))

        model_path = os.path.join(base, 'model_best.pth')
        
        # Save the final results for computing ensemble results
        save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset))

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            evaluation.evalrank(model_path, model=model_without_ddp, split='testall', fold5=True)

            # Evaluate COCO 5K
            evaluation.evalrank(model_path, model=model_without_ddp, split='testall', fold5=False, save_path=save_path)

            if opt.evaluate_cxc:
                # Evaluate COCO-trained models on CxC
                evaluation.evalrank(model_path, model=model_without_ddp, split='testall', fold5=True, cxc=True)

        else:
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, model=model_without_ddp, split='test', fold5=False, save_path=save_path)

        logger.info('Evaluation finish!')    


def train(opt, train_loader, model, model_without_ddp, optimizer, epoch):

    # switch to train mode
    model.train()   

    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    if utils.is_main_process() and epoch == 0:
        logger.info('image encoder trainable parameters: {}M'.format(count_params(model_without_ddp.img_enc)))
        logger.info('txt encoder trainable parameters: {}M'.format(count_params(model_without_ddp.txt_enc)))
        logger.info('criterion trainable parameters: {}M'.format(count_params(model_without_ddp.criterion)))

    n_batch = len(train_loader) 

    end = time.time()

    for i, train_data in enumerate(train_loader):  
        
        optimizer.zero_grad()

        # warmup_alpha is [0, 1], loss = loss * warmup_alpha
        warmup_alpha = float(i) / n_batch if epoch == opt.embedding_warmup_epochs else 1. 

        # measure data loading time
        data_time.update(time.time() - end)

        images, captions, lengths, ids, img_ids = train_data

        # to device
        images = images.cuda(non_blocking=True)
        captions = captions.cuda(non_blocking=True)
        lengths = lengths.cuda(non_blocking=True) 
        img_ids = img_ids.cuda(non_blocking=True) 

        loss = model(images, captions, lengths, img_ids=img_ids, warmup_alpha=warmup_alpha)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.zeros([], requires_grad=True, device=images.device)

        loss.backward()

        if opt.grad_clip > 0:
            clip_grad_norm_(model.parameters(), opt.grad_clip)

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()    

        model_without_ddp.logger = train_logger
        model_without_ddp.logger.update('Iter', model_without_ddp.Eiters)
        model_without_ddp.logger.update('lr', optimizer.param_groups[0]['lr'])   
        model_without_ddp.logger.update('Loss', loss.item(), opt.batch_size)
        model_without_ddp.Eiters += 1

        if utils.is_main_process():
  
            if model_without_ddp.Eiters % opt.log_step == 0:  
                if epoch == opt.embedding_warmup_epochs:
                    logging.info('The first epoch for training backbone, warmup alpha for loss is {}'.format(epoch, warmup_alpha))

                logging.info(
                    'Epoch: [{0}][{1}/{2}]\t'
                    '{e_log}\t'
                    'Batch-Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                        .format(epoch, i+1, n_batch, batch_time=batch_time, e_log=str(model_without_ddp.logger)))

            # Record logs in tensorboard
            tb_logger.log_value('epoch', epoch, step=model_without_ddp.Eiters)
            tb_logger.log_value('step', i, step=model_without_ddp.Eiters)
            tb_logger.log_value('batch_time', batch_time.val, step=model_without_ddp.Eiters)
            tb_logger.log_value('data_time', data_time.val, step=model_without_ddp.Eiters)

            model_without_ddp.logger.tb_log(tb_logger, step=model_without_ddp.Eiters)

        if i > n_batch:
            break


def validate(opt, val_loader, model):

    logger = logging.getLogger(__name__)
    
    model.eval()

    with torch.no_grad():
       img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step, logging.info)

    # have repetitive image features
    img_embs = img_embs[::5]

    start_time = time.time()

    if opt.multi_gpu:         
        sims = torch.zeros((len(img_embs), len(cap_embs))).cuda()
        
        num_tasks = utils.get_world_size()
        rank = utils.get_rank() 

        step = img_embs.size(0) // num_tasks + 1
        start = rank * step
        end = min(img_embs.size(0), start + step)

        sims_part = shard_attn_scores(model, img_embs[start:end], cap_embs, cap_lens, opt, gpu=True)
        sims[start:end] = sims_part

        # wait for synchronization 
        torch.distributed.barrier()
        # Aggregating results on different GPUs
        torch.distributed.all_reduce(sims, op=torch.distributed.ReduceOp.SUM) 
        sims = sims.cpu().numpy()
    else:
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt)      
        sims = sims.numpy()

    # compute metric
    if utils.is_main_process():
        
        logging.info("calculate similarity time: %.3f" % float(time.time() - start_time))

        npts = img_embs.shape[0]
        # print(npts)
        
        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(npts, sims)
        logging.info("Image to text (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1, r5, r10))

        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
        logging.info("Text to image (R@1, R@5, R@10): %.1f, %.1f, %.1f" % (r1i, r5i, r10i))

        # sum of recalls to be used for early stopping
        currscore = r1 + r5 + r10 + r1i + r5i + r10i
        logger.info('Current rsum is {}'.format(round(currscore, 1)))
            
        # record metrics in tensorboard
        tb_logger.log_value('r1', r1, step=model.Eiters)
        tb_logger.log_value('r5', r5, step=model.Eiters)
        tb_logger.log_value('r10', r10, step=model.Eiters)
        tb_logger.log_value('medr', medr, step=model.Eiters)
        tb_logger.log_value('meanr', meanr, step=model.Eiters)
        tb_logger.log_value('r1i', r1i, step=model.Eiters)
        tb_logger.log_value('r5i', r5i, step=model.Eiters)
        tb_logger.log_value('r10i', r10i, step=model.Eiters)
        tb_logger.log_value('medri', medri, step=model.Eiters)
        tb_logger.log_value('meanr', meanr, step=model.Eiters)
        tb_logger.log_value('rsum', currscore, step=model.Eiters)                     

        return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):

    if is_best:
        torch.save(state, os.path.join(prefix, 'model_best.pth'))


def adjust_learning_rate(opt, optimizer, epoch):
    logger = logging.getLogger(__name__)

    decay_rate = opt.decay_rate
    lr_schedules = opt.lr_schedules

    # Sets the learning rate to the initial LR
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by {}'.format(epoch, decay_rate))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * decay_rate
            param_group['lr'] = new_lr
            logger.info('new lr: {}'.format(new_lr))


def count_params(model):

    # The unit is M (million)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    params = round(params/(1024**2), 2)

    return params


if __name__ == '__main__':
    
    main()
