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
from transformers import BertForMaskedLM

from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import grounding_dataset
from dataset.utils import grounding_eval

from refTools.refer_python3 import REFER

from attack import *
from torchvision import transforms
from PIL import Image

def val(model, ref_model, data_loader, tokenizer, device, block_num):
    # test
    model.eval()
    ref_model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print_freq = 50

    images_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    image_attacker = ImageAttacker(config['epsilon'] / 255., preprocess=images_normalize, bounding=(0, 1), cls=args.cls)
    text_attacker = BertAttack(ref_model, tokenizer, cls=args.cls)
    multi_attacker = MultiModalAttacker(model, image_attacker, text_attacker, tokenizer, cls=args.cls)
     
    result = []
    for images, text, ref_ids in metric_logger.log_every(data_loader, print_freq, header):
        images = images.to(device)

        if args.adv != 0:
            images, text = multi_attacker.run_before_fusion(images,text,adv=args.adv, num_iters=config['num_iters'],
                                                            alpha=args.alpha)

        images = images_normalize(images)
        text_input = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True

        image_embeds = model.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(images.device)
        output = model.text_encoder(text_input.input_ids,
                                attention_mask = text_input.attention_mask,
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,
                                return_dict = True,
                               )

        vl_embeddings = output.last_hidden_state[:,0,:]
        vl_output = model.itm_head(vl_embeddings)
        loss = vl_output[:,1].sum()

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)

            grads = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients().detach()
            cams = model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map().detach()

            cams = cams[:, :, :, 1:].reshape(images.size(0), 12, -1, 24, 24) * mask
            grads = grads[:, :, :, 1:].clamp(min=0).reshape(images.size(0), 12, -1, 24, 24) * mask

            gradcam = cams * grads
            gradcam = gradcam.mean(1).mean(1)


        for r_id, cam in zip(ref_ids, gradcam):
            result.append({'ref_id':r_id.item(), 'pred':cam})

        model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = False

    return result


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
    grd_test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')

    test_loader = DataLoader(grd_test_dataset, batch_size=config['batch_size'], num_workers=4)
       
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
        
    ## refcoco evaluation tools
    refer = REFER(config['refcoco_data'], 'refcoco+', 'unc')
    dets = json.load(open(config['det_file'],'r'))
    cocos = json.load(open(config['coco_file'],'r'))    

    #### Model #### 
    print("Creating model")
    model = ALBEF(config = config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    ref_model = BertForMaskedLM.from_pretrained(args.text_encoder)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    #state_dict = checkpoint['model']
    state_dict = checkpoint

    for key in list(state_dict.keys()):
        if 'bert' in key:
            encoder_key = key.replace('bert.','')
            state_dict[encoder_key] = state_dict[key]
            del state_dict[key]
    msg = model.load_state_dict(state_dict,strict=False)

    print('load checkpoint from %s'%args.checkpoint)
    #print(msg)
    
    model = model.to(device)
    ref_model = ref_model.to(device)
    

    print("Start Evaluating")
    start_time = time.time()    

    result = val(model, ref_model, test_loader, tokenizer, device, args.block_num)

    grounding_acc = grounding_eval(result, dets, cocos, refer, alpha=0.5, mask_size=24)

    log_stats = {**{f'{k}': v for k, v in grounding_acc.items()},
                 'eval type': args.adv, 'cls': args.cls, 'eps': config['epsilon'], 'iters':config['num_iters'], 'alpha': args.alpha
                }

    with open(os.path.join(args.output_dir, "log.txt"),"a+") as f:
        f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluating time {}'.format(total_time_str))

            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Grounding.yaml')
    parser.add_argument('--checkpoint', default='checkpoints/refcoco.pth')
    parser.add_argument('--output_dir', default='output/VG')
    parser.add_argument('--block_num', default=8, type=int)
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--adv', default=0, type=int,
                        help='0=clean, 1=adv text, 2=adv image, 3=adv text and adv image,')
    parser.add_argument('--cls', action='store_true')
    parser.add_argument('--alpha', default=3.0, type=float)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
