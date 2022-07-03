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
    num_text = len(data_loader.dataset.text)
    num_image = len(data_loader.dataset.ann)

    #image_feats = torch.zeros(num_image, config['embed_dim'])
    image_feats = torch.zeros(num_image, model.visual.output_dim)

    #text_feats = torch.zeros(num_text, config['embed_dim'])
    text_feats = torch.zeros(num_text, model.visual.output_dim)
    #text_atts = torch.zeros(num_text, 30).long()

    print('Forward')
    for images, texts, texts_ids in data_loader:
        images = images.to(device)
        if args.adv != 0:
            images, texts = multi_attacker.run_before_fusion(images, texts, adv=args.adv, num_iters=config['num_iters'], max_length=77,
                                                             alpha=args.alpha)

        images_ids = [data_loader.dataset.txt2img[i.item()] for i in texts_ids]
        with torch.no_grad():
            images = images_normalize(images)
            output = model.inference(images, texts)

            image_feats[images_ids] = output['image_feat'].cpu().float().detach()
            text_feats[texts_ids] = output['text_feat'].cpu().float().detach()

    sims_matrix = image_feats @ text_feats.t()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return sims_matrix.cpu().numpy(), sims_matrix.t().cpu().numpy()


@torch.no_grad()
def retrieval_score(model,  image_embeds, text_embeds, text_atts, num_image, num_text, device=None):
    if device is None:
        device = image_embeds.device

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation Direction Similarity With Bert Attack:'

    sims_matrix = F.normalize(image_embeds, dim=-1) @ F.normalize(text_embeds, dim=-1).t()
    score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)

        encoder_output = image_embeds[i].repeat(config['k_test'], 1, 1).to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_embeds[topk_idx].to(device),
                                    attention_mask=text_atts[topk_idx].to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_i2t[i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix, 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.text_encoder(encoder_embeds=text_embeds[i].repeat(config['k_test'], 1, 1).to(device),
                                    attention_mask=text_atts[i].repeat(config['k_test'], 1).to(device),
                                    encoder_hidden_states=encoder_output,
                                    encoder_attention_mask=encoder_att,
                                    return_dict=True,
                                    mode='fusion'
                                    )
        score = model.itm_head(output.last_hidden_state[:, 0, :])[:, 1]
        score_matrix_t2i[i, topk_idx] = score

    return score_matrix_i2t, score_matrix_t2i


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, img2txt, txt2img):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result

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

    score_i2t, score_t2i = retrieval_eval(model, ref_model, test_loader, tokenizer, device, config)

    result = itm_eval(score_i2t, score_t2i, test_dataset.img2txt, test_dataset.txt2img)
    print(result)
    log_stats = {**{f'test_{k}': v for k, v in result.items()},
                 'eval type': args.adv, 'cls':args.cls, 'eps': config['epsilon'], 'iters':config['num_iters'], 'alpha': args.alpha}
    with open(os.path.join(args.output_dir, "log_CLIP.txt"), "a+") as f:
        f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluate time {}'.format(total_time_str))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_coco.yaml')
    parser.add_argument('--output_dir', default='output/retrieval')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--image_encoder', default='ViT-B/16')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--adv', default=0, type=int,
                        help='0=clean, 1=adv text, 2=adv image, 3=adv text and adv image,')
    parser.add_argument('--alpha', default=3.0, type=float)
    parser.add_argument('--cls', action='store_true')

    args = parser.parse_args()

    # the output of CLIP is [CLS] embedding, so needn't to select at 0
    args.cls = False
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)

