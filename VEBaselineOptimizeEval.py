import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from transformers import BertForMaskedLM

from models.model_ve import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import ve_dataset
from PIL import Image
from torchvision import transforms


class Attacker():
    def __init__(self, lr, B, lam_1, lam_2, *args, **kwargs):
        self.lr = lr
        self.tau = np.log2(3)
        self.B = B
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.preprocess = kwargs.get('preprocess')
        self.bounding = kwargs.get('bounding')
        if self.bounding is None:
            self.bounding = (0, 1)

    def attack(self, net, images, text_inputs, labels, num_iters, epsilon):
        self.delta = torch.zeros_like(images)
        self.delta.data.uniform_(-self.B, self.B)
        self.delta.requires_grad = True

        optimizer = torch.optim.Adam([self.delta], lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        N_sqrt = images.size(0) ** 0.5
        for i in range(num_iters):
            image_diversity = images + self.delta

            if self.preprocess is not None:
                image_diversity = self.preprocess(image_diversity)

            predict = net(image_diversity, text_inputs, train=False)
            loss_1 = self.lam_1 * (self.tau - criterion(predict, labels))
            loss_2 = self.lam_2 * torch.relu(torch.norm(self.delta) / (N_sqrt) - (self.B - 2)/255.0)
            loss = loss_1+loss_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #self.delta.data = torch.clamp(self.delta, -epsilon, epsilon).data

        return torch.clamp((images + self.delta), *self.bounding).detach()


def evaluate(model, ref_model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    ref_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    attacker = Attacker(lr=1.0, B=20, lam_1=1, lam_2=10, preprocess=images_normalize)


    for images, text, targets in metric_logger.log_every(data_loader, print_freq, header):
        images, targets = images.to(device), targets.to(device)


        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        images = attacker.attack(model, images, text_inputs, targets, 50, epsilon=config['epsilon'])

        images = images_normalize(images)

        with torch.no_grad():
            prediction = model(images, text_inputs, targets=targets, train=False)

        _, pred_class = prediction.max(1)
        accuracy = (targets == pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=images.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    device = args.gpu[0]

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
    ])
    datasets = ve_dataset(config['test_file'], test_transform, config['image_root'])
    test_loader = DataLoader(datasets, batch_size=config['batch_size_test'], num_workers=4, shuffle=True)

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['model']

    # reshape positional embedding to accomodate for image resolution change
    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s' % args.checkpoint)
    #print(msg)

    model = model.to(device)
    ref_model = ref_model.to(device)


    print("Start evaluating")
    start_time = time.time()

    test_stats = evaluate(model, ref_model, test_loader, tokenizer, device, config)

    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'eval type': 'optimize'}

    with open(os.path.join(args.output_dir, "log.txt"), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VE.yaml')
    parser.add_argument('--output_dir', default='output_baseline/VE')
    parser.add_argument('--checkpoint', default='checkpoints/VE.pth')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--gpu', type=int, nargs='+',  default=[0])
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
