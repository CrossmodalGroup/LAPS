from __future__ import print_function

import logging
import time
import torch
import numpy as np
import sys
from collections import OrderedDict
from transformers import BertTokenizer

from lib import utils
from lib import image_caption
from lib.vse import VSEModel


logger = logging.getLogger(__name__)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # current values
        self.val = val
        # total values
        self.sum += val * n
        # the number of records
        self.count += n
        # average values
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):

        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=logger.info):

    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    
    # compute the number of max word
    max_n_word = model.opt.max_word

    for i, data_i in enumerate(data_loader):
        
        # make sure val logger is used       
        images, captions, lengths, ids, img_ids = data_i

        model.logger = val_logger

        # compute the embeddings
        img_emb, cap_emb, lengths = model.forward_emb(images, captions, lengths)

        if img_embs is None:
            # for local visual features
            img_embs = torch.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            # for local textual features
            cap_embs = torch.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            
            cap_lens = torch.zeros(len(data_loader.dataset)).long()

        # cache embeddings
        img_embs[ids] = img_emb.cpu()

        n_word = min(max(lengths), max_n_word)
        
        cap_embs[ids, :n_word, :] = cap_emb[:, :n_word, :].cpu()
        cap_lens[ids] = lengths.cpu()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Batch-Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(i, len(data_loader.dataset) // data_loader.batch_size + 1, batch_time=batch_time, e_log=str(model.logger)))
        del images, captions

    return img_embs, cap_embs, cap_lens


def evalrank(model_path, model=None, data_path=None, split='dev', fold5=False, save_path=None):

    # load model and options
    checkpoint = torch.load(model_path, map_location='cuda')
    opt = checkpoint['opt']
    
    opt.dataset = 'coco' if split == 'testall' else 'f30k'

    logger.info(opt)

    # load vocabulary used by the model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer = BertTokenizer.from_pretrained(opt.bert_path)

    # construct model
    if model is None:
        model = VSEModel(opt).cuda()

    # load model state
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()

    logger.info('Loading dataset')
    data_loader = image_caption.get_test_loader(opt, data_path, tokenizer, 128, opt.workers, split)

    logger.info('Computing results...')
    with torch.no_grad():
        img_embs, cap_embs, cap_lens = encode_data(model, data_loader)

    # one image to five captions, since have repetitive images
    logger.info('Images: %d, Captions: %d' % (img_embs.shape[0] / 5, cap_embs.shape[0]))

    # for F30K, imgs 1000, captions 5000.
    # for COCO, imgs 5000, captions 25000. (5-fold is five times of 1000 imgs)

    if not fold5:
        img_embs = img_embs[::5]
        
        start = time.time()
        sims = shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt).numpy()
        end = time.time()

        # npts = the number of images
        npts = img_embs.shape[0]

        if save_path is not None:
            np.save(save_path, {'npts': npts, 'sims': sims})
            logger.info('Save the similarity into {}'.format(save_path))

        logger.info("calculate similarity time: {}".format(end - start))

        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)

        # r[0] -> R@1, r[1] -> R@5, r[2] -> R@10
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3

        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        logger.info("rsum: %.1f" % rsum)
        # logger.info("Average i2t Recall: %.1f" % ar)
        logger.info("Image to text (R@1, R@5, R@10): %.1f %.1f %.1f" % r[:3])
        # logger.info("Average t2i Recall: %.1f" % ari)
        logger.info("Text to image (R@1, R@5, R@10): %.1f %.1f %.1f" % ri[:3])
    
    else:
        # 5 fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            
            start = time.time()
            sims = shard_attn_scores(model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt).numpy()
            end = time.time()

            logger.info("calculate similarity time: {}".format(end - start))

            npts = img_embs_shard.shape[0]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            ri, rti0 = t2i(npts, sims, return_ranks=True)

            logger.info("Image to text: %.1f, %.1f, %.1f" % r[:3])
            logger.info("Text to image: %.1f, %.1f, %.1f" % ri[:3])

            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            # logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        logger.info("-----------------------------------")
        logger.info("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        logger.info("rsum: %.1f" % (mean_metrics[12]))
        # logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
        logger.info("Image to text (R@1, R@5, R@10): %.1f %.1f %.1f" % mean_metrics[:3])
        # logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
        logger.info("Text to image (R@1, R@5, R@10): %.1f %.1f %.1f" % mean_metrics[5:8])


def i2t(npts, sims, return_ranks=False, mode='coco'):

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        
        inds = np.argsort(sims[index])[::-1]

        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, return_ranks=False, mode='coco'):

    if mode == 'coco':
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode == 'coco':
            for i in range(5):
                inds = np.argsort(sims[5 * index + i])[::-1]
                ranks[5 * index + i] = np.where(inds == index)[0][0]
                top1[5 * index + i] = inds[0]
        else:
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, gpu=False):

    shard_size = opt.shard_size
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = torch.zeros((len(img_embs), len(cap_embs)))
    if gpu:
        sims = sims.cuda()
    
    with torch.no_grad(): 
        
        for i in range(n_im_shard):    
            
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))

            for j in range(n_cap_shard):

                if utils.is_main_process():
                    sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))

                ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))
                 
                im = img_embs[im_start:im_end].cuda()
                ca = cap_embs[ca_start:ca_end].cuda()
                l = cap_lens[ca_start:ca_end].long().cuda()

                sim = model.forward_sim(im, ca, l)
                if not gpu:
                    sim = sim.cpu()

                sims[im_start:im_end, ca_start:ca_end] = sim

    return sims


if __name__ == '__main__':

    pass
