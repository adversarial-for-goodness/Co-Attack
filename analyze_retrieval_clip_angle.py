import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models import clip
from models.tokenization_bert import BertTokenizer

from transformers import BertForMaskedLM

import utils

from attack import *
from torchvision import transforms

from dataset import pair_dataset
from PIL import Image
from torchvision import transforms

import seaborn as sns
import matplotlib.pyplot as plt


def retrieval_eval(model, ref_model, data_loader, tokenizer, device, config):
    # test
    model.float()
    model.eval()
    ref_model.eval()

    print('Computing features for evaluation adv...')
    start_time = time.time()

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    image_attacker = ImageAttacker(config['epsilon'] / 255., preprocess=images_normalize, bounding=(0, 1), cls=args.cls)
    text_attacker = BertAttack(ref_model, tokenizer, cls=args.cls)
    multi_attacker = MultiModalAttacker(model, image_attacker, text_attacker, tokenizer, cls=args.cls)

    print('Prepare memory')

    angle = []

    print('Forward')
    for images, text, texts_ids in data_loader:
        images = images.to(device)

        images_adv, text_adv = multi_attacker.run_before_fusion(images, text, adv=args.adv, num_iters=config['num_iters'])
        with torch.no_grad():
            images = images_normalize(images)
            origin_output = model.inference(images, text)

            images_adv = images_normalize(images_adv)
            adv_output = model.inference(images_adv, text_adv)


        with torch.no_grad():
            dir_image = F.normalize(adv_output['image_feat'] - origin_output['image_feat'], dim=-1)
            dir_text = F.normalize(adv_output['text_feat'] - origin_output['text_feat'], dim=-1)
            angle_batch = (dir_image * dir_text).sum(dim=-1)
            angle.append(angle_batch.cpu().detach())

    angle = torch.cat(angle, dim=0).numpy()
    print('angle mean: {}'.format(angle.mean()))
    try:
        with open(os.path.join(args.output_dir, 'log.txt'), 'a+') as f:
            f.write(json.dumps({'file': args.file, 'mean': angle.mean().item()}))
    except:
        pass
    sns.displot(angle)
    plt.savefig(os.path.join(args.output_dir, args.file + '.png'))
    torch.save(angle, os.path.join(args.output_dir, args.file + '.pth'))



def main(args, config):
    device = args.gpu[0]

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model, preprocess = clip.load(args.image_encoder, device=device)
    model.set_tokenizer(tokenizer)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)

    model = model.to(device)
    ref_model = ref_model.to(device)

    #### Dataset ####
    print("Creating dataset")
    n_px = model.visual.input_resolution
    test_transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=Image.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
    ])
    test_dataset = pair_dataset(config['test_file'], test_transform, config['image_root'])

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], num_workers=4)


    print("Start eval")
    start_time = time.time()

    retrieval_eval(model, ref_model, test_loader, tokenizer, device, config)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_coco.yaml')
    parser.add_argument('--output_dir', default='output/retrieval')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--image_encoder', default='ViT-L/14')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--adv', default=0, type=int,
                        help='0=clean, 1=adv text, 2=adv image, 3=adv text and adv image,')
    parser.add_argument('--file', default='clip_angle', type=str)
    parser.add_argument('--cls', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

