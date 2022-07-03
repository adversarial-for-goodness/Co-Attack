import torch
import torch.nn.functional as F
from enum import Enum
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

class NormType(Enum):
    Linf = 0
    L2 = 1

def clamp_by_l2(x, max_norm):
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor

def random_init(x, norm_type, epsilon):
    delta = torch.zeros_like(x)
    if norm_type == NormType.Linf:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == NormType.L2:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel


class ImageAttacker():
    # PGD
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, cls=True, *args, **kwargs):
        self.norm_type = norm_type
        self.random_init = random_init
        self.epsilon = epsilon
        self.cls = cls
        self.preprocess = kwargs.get('preprocess')
        self.bounding = kwargs.get('bounding')
        if self.bounding is None:
            self.bounding = (0, 1)

    def input_diversity(self, image):
        return image

    def attack(self, image, num_iters):
        if self.random_init:
            self.delta = random_init(image, self.norm_type, self.epsilon)
        else:
            self.delta = torch.zeros_like(image)

        if hasattr(self, 'kernel'):
            self.kernel = self.kernel.to(image.device)

        if hasattr(self, 'grad'):
            self.grad = torch.zeros_like(image)


        epsilon_per_iter = self.epsilon / num_iters * 1.25

        for i in range(num_iters):
            self.delta = self.delta.detach()
            self.delta.requires_grad = True

            image_diversity = self.input_diversity(image + self.delta)
            #plt.imshow(image_diversity.cpu().detach().numpy()[0].transpose(1, 2, 0))
            if self.preprocess is not None:
                image_diversity = self.preprocess(image_diversity)

            yield image_diversity

            grad = self.get_grad()
            grad = self.normalize(grad)
            self.delta = self.delta.data + epsilon_per_iter * grad

            # constraint 1: epsilon
            self.delta = self.project(self.delta, self.epsilon)
            # constraint 2: image range
            self.delta = torch.clamp(image + self.delta, *self.bounding) - image

        yield (image + self.delta).detach()

    def get_grad(self):
        self.grad = self.delta.grad.clone()
        return self.grad

    def project(self, delta, epsilon):
        if self.norm_type == NormType.Linf:
            return torch.clamp(delta, -epsilon, epsilon)
        elif self.norm_type == NormType.L2:
            return clamp_by_l2(delta, epsilon)

    def normalize(self, grad):
        if self.norm_type == NormType.Linf:
            return torch.sign(grad)
        elif self.norm_type == NormType.L2:
            return grad / torch.norm(grad, dim=(1, 2, 3), p=2, keepdim=True)

    def run_trades(self, net, image, num_iters):
        with torch.no_grad():
            origin_output = net.inference_image(self.preprocess(image))
            if self.cls:
                origin_embed = origin_output['image_embed'][:, 0, :].detach()
            else:
                origin_embed = origin_output['image_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        attacker = self.attack(image, num_iters)

        for i in range(num_iters):
            image_adv = next(attacker)
            adv_output = net.inference_image(image_adv)
            if self.cls:
                adv_embed = adv_output['image_embed'][:, 0, :]
            else:
                adv_embed = adv_output['image_embed'].flatten(1)

            loss = criterion(adv_embed.log_softmax(dim=-1), origin_embed.softmax(dim=-1))
            loss.backward()

        image_adv = next(attacker)
        return image_adv


class ImageAttack_DI(ImageAttacker):
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, cls=True, resize_rate=1.10, diversity_prob=0.3, *args, **kwargs):
        super(ImageAttack_DI, self).__init__(epsilon, norm_type, random_init, cls, *args, **kwargs)
        self.resize_rate = resize_rate
        self.diversity_prob = diversity_prob

    def input_diversity(self, x):
        assert self.resize_rate >= 1.0
        assert self.diversity_prob >= 0.0 and self.diversity_prob <= 1.0

        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)
        # print(img_size, img_resize, resize_rate)
        rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
        padded = F.interpolate(padded, size=[img_size, img_size])
        ret = padded if torch.rand(1) < self.diversity_prob else x
        return ret


class ImageAttack_MI(ImageAttacker):
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, cls=True, momentum=0.9, *args, **kwargs):
        super(ImageAttack_MI, self).__init__(epsilon, norm_type, random_init, cls, *args, **kwargs)
        self.momentum = momentum

    def get_grad(self):
        if not hasattr(self, 'grad'):
            self.grad = torch.zeros_like(self.delta)

        grad = self.delta.grad.clone()
        self.grad = self.grad * self.momentum + grad
        return self.grad


class ImageAttack_DIM(ImageAttack_DI, ImageAttack_MI):
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, cls=True, momentum=0.9, resize_rate=1.10, diversity_prob=0.3, *args, **kwargs):
        super(ImageAttack_DIM, self).__init__(epsilon, norm_type, random_init, cls, resize_rate, diversity_prob, momentum, *args, **kwargs)

    def input_diversity(self, x):
        return ImageAttack_DI.input_diversity(self, x)

    def get_grad(self):
        if not hasattr(self, 'grad'):
            self.grad = torch.zeros_like(self.delta)

        grad = self.delta.grad.clone()
        self.grad = self.grad * self.momentum + grad
        return self.grad


class ADMIX_Attack_DIM(ImageAttack_DIM):
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, cls=True,
                 portion=0.2, repeat=3, resize_rate=1.10, diversity_prob=0.3, momentum=0.9, *args, **kwargs):
        super(ADMIX_Attack_DIM, self).__init__(epsilon, norm_type, random_init, cls,
                                               momentum, resize_rate, diversity_prob,  *args, **kwargs)
        self.portion = portion
        self.repeat = repeat

    def input_diversity(self, x):
        x = ImageAttack_DIM.input_diversity(self, x)
        x = torch.cat([(x + self.portion * x[torch.randperm(x.shape[0])]) for _ in range(self.repeat)], dim=0)
        return x

class ImageAttack_SI(ImageAttacker):
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, cls=True, repeat=5, *args, **kwargs):
        super(ImageAttack_SI, self).__init__(epsilon, norm_type, random_init, cls, *args, **kwargs)
        self.repeat = repeat

    def input_diversity(self, x):
        x = ImageAttacker.input_diversity(self, x)
        x_repeat = []
        for i in range(self.repeat):
            x_repeat.append(x * 0.5**i)
        return torch.cat(x_repeat, dim=0)

    def run_trades(self, net, image, num_iters):
        with torch.no_grad():
            origin_output = net.inference_image(self.preprocess(image.repeat(self.repeat, 1, 1, 1)))
            if self.cls:
                origin_embed = origin_output['image_embed'][:, 0, :].detach()
            else:
                origin_embed = origin_output['image_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        attacker = self.attack(image, num_iters)

        for i in range(num_iters):
            image_adv = next(attacker)
            adv_output = net.inference_image(image_adv)
            if self.cls:
                adv_embed = adv_output['image_embed'][:, 0, :]
            else:
                adv_embed = adv_output['image_embed'].flatten(1)

            loss = criterion(adv_embed.log_softmax(dim=-1), origin_embed.softmax(dim=-1))
            loss.backward()

        image_adv = next(attacker)
        return image_adv
