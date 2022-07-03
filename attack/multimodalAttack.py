import torch
from transformers import BatchEncoding
import torch.nn.functional as F

def equal_normalize(x):
    return x

class MultiModalAttacker():
    def __init__(self, net, image_attacker, text_attacker, tokenizer, cls=True, *args, **kwargs):
        self.net = net
        self.image_attacker = image_attacker
        self.text_attacker = text_attacker
        self.tokenizer = tokenizer
        self.cls = cls
        if hasattr(text_attacker, 'sample_batch_size'):
            self.sample_batch_size = text_attacker.sample_batch_size
        if hasattr(image_attacker, 'preprocess'):
            self.image_normalize = image_attacker.preprocess
        else:
            self.image_normalize = equal_normalize

        self.repeat = 1
        if hasattr(image_attacker, 'repeat'):
            self.repeat = image_attacker.repeat


    def run(self, images, text, adv, num_iters=10, k=10, max_length=30, alpha=3.0):
        with torch.no_grad():
            device = images.device
            text_input = self.tokenizer(text * self.repeat, padding='max_length', truncation=True, max_length=max_length,
                                        return_tensors="pt").to(device)
            origin_output = self.net.inference(self.image_normalize(images).repeat(self.repeat, 1, 1,1),
                                               text_input, use_embeds=False)
            if self.cls:
                origin_embeds = origin_output['fusion_output'][:, 0, :].detach()
            else:
                origin_embeds = origin_output['fusion_output'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='batchmean')

        # image
        if adv == 2 or adv == 3:
            image_attack = self.image_attacker.attack(images, num_iters)
            for i in range(num_iters):
                image_diversity = next(image_attack)
                adv_output = self.net.inference(image_diversity, text_input, use_embeds=False)
                if self.cls:
                    adv_embed = adv_output['fusion_output'][:, 0, :]
                else:
                    adv_embed = adv_output['fusion_output'].flatten(1)
                loss = criterion(adv_embed.log_softmax(dim=-1), origin_embeds.softmax(dim=-1))
                loss.backward()
            images_adv = next(image_attack)

        elif adv == 4:
            image_attack = self.image_attacker.attack(images, num_iters)
            with torch.no_grad():
                text_adv = self.text_attacker.attack(self.net, images, text, k)
                text_input = self.tokenizer(text_adv * self.repeat, padding='max_length', truncation=True, max_length=max_length,
                                            return_tensors="pt").to(device)
                text_adv_output = self.net.inference(self.image_normalize(images).repeat(self.repeat, 1, 1, 1),
                                                     text_input, use_embeds=False)
                if self.cls:
                    text_adv_embed = text_adv_output['fusion_output'][:, 0, :].detach()
                else:
                    text_adv_embed = text_adv_output['fusion_output'].flatten(1).detach()

            for i in range(num_iters):
                image_diversity = next(image_attack)
                adv_output = self.net.inference(image_diversity, text_input, use_embeds=False)
                if self.cls:
                    adv_embed = adv_output['fusion_output'][:, 0, :]
                else:
                    adv_embed = adv_output['fusion_output'].flatten(1)
                loss_clean_text = criterion(adv_embed.log_softmax(dim=-1), origin_embeds.softmax(dim=-1))

                loss_adv_text = criterion(adv_embed.log_softmax(dim=-1), text_adv_embed.softmax(dim=-1))
                loss = loss_adv_text + alpha * loss_clean_text
                loss.backward()
            images_adv = next(image_attack)

        else:
            images_adv = images

        # text
        if adv == 1 or adv == 3 or adv == 4 or adv == 5:
            with torch.no_grad():
                text_adv = self.text_attacker.attack(self.net, images, text, k)
        else:
            text_adv = text

        return images_adv, text_adv


    def run_before_fusion(self, images, text, adv, num_iters=10, k=10, max_length=30, alpha=3.0):
        if adv == 2 or adv == 3:
            images_adv = self.image_attacker.run_trades(self.net, images, num_iters)
        elif adv == 4:
            device = images.device
            image_attack = self.image_attacker.attack(images, num_iters)
            criterion = torch.nn.KLDivLoss(reduction='batchmean')

            with torch.no_grad():
                text_adv = self.text_attacker.attack(self.net, text, k)
                text_input = self.tokenizer(text_adv, padding='max_length', truncation=True, max_length=max_length,
                                            return_tensors="pt").to(device)
                text_adv_output = self.net.inference_text(text_input)

                if self.cls:
                    text_adv_embed = text_adv_output['text_embed'][:, 0, :].detach()
                else:
                    text_adv_embed = text_adv_output['text_embed'].flatten(1).detach()

            with torch.no_grad():
                image_output = self.net.inference_image(self.image_normalize(images))
                if self.cls:
                    image_embed = image_output['image_embed'][:, 0, :].detach()
                else:
                    image_embed = image_output['image_embed'].flatten(1).detach()

            for i in range(num_iters):
                image_adv = next(image_attack)
                image_adv_output = self.net.inference_image(image_adv)
                if self.cls:
                    image_adv_embed = image_adv_output['image_embed'][:, 0, :]
                else:
                    image_adv_embed = image_adv_output['image_embed'].flatten(1)

                loss_image_trades = criterion(image_adv_embed.log_softmax(dim=-1), image_embed.softmax(dim=-1).repeat(self.repeat, 1))

                loss_adv_text = criterion(F.normalize(image_adv_embed, dim=-1).log_softmax(dim=-1),
                                          F.normalize(text_adv_embed, dim=-1).softmax(dim=-1).repeat(self.repeat, 1))
                loss = loss_image_trades + alpha * loss_adv_text
                loss.backward()

            images_adv = next(image_attack)

        else:
            images_adv = images

        if adv == 1 or adv == 3 or adv == 4 or adv == 5:
            with torch.no_grad():
                text_adv = self.text_attacker.attack(self.net, text, k)
        else:
            text_adv = text

        return images_adv, text_adv


class MultiModalAttackerClassify():
    def __init__(self, net, image_attacker, text_attacker, tokenizer, cls=True, *args, **kwargs):
        self.net = net
        self.image_attacker = image_attacker
        self.text_attacker = text_attacker
        self.tokenizer = tokenizer
        self.cls = cls
        self.repeat = 1
        if hasattr(text_attacker, 'sample_batch_size'):
            self.sample_batch_size = text_attacker.sample_batch_size
        if hasattr(image_attacker, 'preprocess'):
            self.image_normalize = image_attacker.preprocess
        else:
            self.image_normalize = equal_normalize

        if hasattr(image_attacker, 'repeat'):
            self.repeat = image_attacker.repeat

    def run(self, images, text, labels, adv, num_iters=10, k=10, max_length=30, alpha=3):
        device = images.device

        text_input = self.tokenizer(text * self.repeat, padding='max_length', truncation=True, max_length=max_length,
                                    return_tensors="pt").to(device)
        labels = labels.repeat(self.repeat)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        # image
        if adv == 2 or adv == 3:
            image_attack = self.image_attacker.attack(images, num_iters)
            for i in range(num_iters):
                image_diversity = next(image_attack)
                adv_output = self.net.inference(image_diversity, text_input, use_embeds=False)

                predict = adv_output['predict']

                loss = criterion(predict, labels)
                loss.backward()
            images_adv = next(image_attack)

        elif adv == 4:
            image_attack = self.image_attacker.attack(images, num_iters)
            with torch.no_grad():
                text_adv = self.text_attacker.attack(self.net, images, text, k)
                text_input = self.tokenizer(text_adv * self.repeat, padding='max_length', truncation=True, max_length=max_length,
                                            return_tensors="pt").to(device)
                text_adv_output = self.net.inference(self.image_normalize(images).repeat(self.repeat, 1, 1, 1),
                                                     text_input, use_embeds=False)

                text_adv_embed = text_adv_output['fusion_output'].flatten(1).detach()


            for i in range(num_iters):
                image_diversity = next(image_attack)
                adv_output = self.net.inference(image_diversity, text_input, use_embeds=False)

                predict = adv_output['predict']
                adv_embed = adv_output['fusion_output'].flatten(1)
                loss_clean_text = criterion(predict, labels)

                loss_adv_text = kl_criterion(adv_embed.log_softmax(dim=-1), text_adv_embed.softmax(dim=-1))
                loss = loss_adv_text + alpha * loss_clean_text
                loss.backward()
            images_adv = next(image_attack)

        else:
            images_adv = images

        # text
        if adv == 1 or adv == 3 or adv == 4 or adv == 5:
            with torch.no_grad():
                text_adv = self.text_attacker.attack(self.net, images, text, k)
        else:
            text_adv = text

        return images_adv, text_adv
