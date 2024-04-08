import os
import torch
import argparse
import logging
from lib import evaluation


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='f30k', help='coco or f30k')
    parser.add_argument('--data_path', type=str, default='data/', help='the path of dataset')
    parser.add_argument('--save_results', type=int, default=0, help='whether save the results')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu id')

    opt = parser.parse_args()

    torch.cuda.set_device(opt.gpu_id)

    if opt.dataset == 'coco':
        weights_bases = [    
            'runs/coco_vit',
            'runs/coco_swin',
        ]
    else:
        weights_bases = [
            'runs/f30k_vit',
            'runs/f30k_swin',
        ]

    for base in weights_bases:

        # logging.basicConfig(filename=os.path.join(base, 'eval_extra.log'), filemode='w', 
        #                     format='%(asctime)s %(message)s', level=logging.DEBUG, force=True)
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        logger.info('Evaluating {}...'.format(base))
        model_path = os.path.join(base, 'model_best.pth')
        
        # Save the final results for computing ensemble results
        save_path = os.path.join(base, 'results_{}.npy'.format(opt.dataset)) if opt.save_results else None

        if opt.dataset == 'coco':
            # Evaluate COCO 5-fold 1K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=True)
            # Evaluate COCO 5K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='testall', fold5=False, save_path=save_path)
        else:
            # Evaluate Flickr30K
            evaluation.evalrank(model_path, data_path=opt.data_path, split='test', fold5=False, save_path=save_path)


if __name__ == '__main__':
    
    main()
