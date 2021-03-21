from imagenetv2_pytorch import ImageNetV2Dataset
import torch
import torchvision
from torchvision import transforms
import PIL
import matplotlib.pyplot as plt
import numpy as np

img_path = './dog.JPEG'

img = PIL.Image.open(img_path)
big_dim = max(img.width, img.height)
wide = img.width > img.height
new_w = 299 if not wide else int(img.width * 299. / float(img.height))
new_h = 299 if wide else int(img.height * 299. / float(img.width))
img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))

x = torch.from_numpy(np.array(img, dtype=float)) / 255.
x = x.unsqueeze(0).to('cuda', dtype=torch.float)
x = torch.transpose(x, 3,1)

inception = torchvision.models.inception_v3(pretrained=True)
inception.cuda()
inception.eval()

import ast
file = open("imagenet1000_clsidx_to_labels.txt", "r")
contents = file.read()
label_names = ast.literal_eval(contents)
file.close()

def plot_prediction(inp):
  plt.imshow(torch.transpose(inp.cpu(), 1, 3).squeeze().numpy())
  with torch.no_grad():
    plt.xlabel(f"inception's prediction: {label_names[inception(inp).argmax().cpu().item()]}", fontsize=15)
plot_prediction(x)

# Our target class will be a ping-pong ball:
target_label = 722
target_name = label_names[target_label]
target_y = torch.Tensor([target_label]).long()
print(target_name)

import torch.nn as nn


def grad(model, inputs, y, dev='cuda', loss_func=nn.CrossEntropyLoss(reduction='sum')):
    y = y.to(dev)
    x = inputs.detach().to(dev).requires_grad_(True)

    model.to(dev)
    model.zero_grad()
    output = model(x)

    loss = loss_func(output, y)
    loss.backward()

    g = x.grad.detach()
    return g


def grad_v2(model, inputs, y, dev='cuda', loss_func=nn.CrossEntropyLoss(reduction='sum')):
    y = y.to(dev)
    x = inputs.detach().to(dev).requires_grad_(True)

    num_attacks=10

    model.to(dev)
    model.zero_grad()
    outputs = model(xs)

    loss = loss_func(outputs, ys)
    loss.backward()

    g = x.grad.detach()
    return g

def pgd_attack(targ_model, X, Y, grad_fn=grad, num_steps=10, eps=0.03):
    X = X.cuda()
    Y = Y.cuda()

    if eps == 0.:
        return X.detach()

    alpha = eps / np.sqrt(num_steps)

    def proj(z, maxnorm):
        return torch.clamp(z, -eps, eps)

    def normalize(z):
        return torch.sign(z)

    def nan_to_zero(z):
        return torch.where(torch.isnan(z), torch.zeros_like(z), z)  # nan --> 0

    delta = torch.zeros_like(X).cuda()

    for _ in range(num_steps):
        g = grad_fn(targ_model, X + delta, Y)
        g = nan_to_zero(g)

        with torch.no_grad():
            d_step = -alpha * normalize(g)
            delta = delta + d_step
            delta = proj(delta, maxnorm=eps)
            delta = delta.detach()

    x_adv = X + delta
    return x_adv.detach()


def better_pgd_attack_v2(sigma,n_attack_samples, targ_model, X, Y, grad_fn=grad, num_steps=10, eps=0.03):
    X = X.cuda()
    Y = Y.cuda()

    if eps == 0.:
        return X.detach()

    alpha = eps / np.sqrt(num_steps)

    def proj(z, maxnorm):
        return torch.clamp(z, -eps, eps)

    def normalize(z):
        return torch.sign(z)

    def nan_to_zero(z):
        return torch.where(torch.isnan(z), torch.zeros_like(z), z)  # nan --> 0

    Xs = X
    Ys= Y
    for i in range(n_attack_samples-1):

        noise = torch.empty(size=X.shape).normal_(mean=0, std=sigma).to('cuda', dtype=torch.float)

        X_noise = X + noise
        Xs = torch.cat((Xs,X_noise),0)
        Ys = torch.cat((Ys,Y),0)
        print(Xs.shape)
    delta = torch.zeros_like(X).cuda()

    for _ in range(num_steps):
        g = grad_fn(targ_model, Xs + delta, Ys)
        g = nan_to_zero(g)

        with torch.no_grad():
            d_step = -alpha * normalize(g)
            delta = delta + d_step
            delta = proj(delta, maxnorm=eps)
            delta = delta.detach()

    x_adv = X + delta
    return x_adv.detach()

def better_pgd_attack(sigma,n_attack_samples, targ_model, X, Y, grad_fn=grad, num_steps=10, eps=0.03):
    X = X.cuda()
    Y = Y.cuda()

    if eps == 0.:
        return X.detach()

    alpha = eps / np.sqrt(num_steps)

    def proj(z, maxnorm):
        return torch.clamp(z, -eps, eps)

    def normalize(z):
        return torch.sign(z)

    def nan_to_zero(z):
        return torch.where(torch.isnan(z), torch.zeros_like(z), z)  # nan --> 0

    delta = torch.zeros_like(X).cuda()

    delta_sum = torch.zeros_like(X).cuda()

    for i in range(n_attack_samples):
        noise = torch.empty(size=X.shape).normal_(mean=0, std=sigma).to('cuda', dtype=torch.float)

        X_noise = X + noise
        for _ in range(num_steps):
            g = grad_fn(targ_model, X_noise + delta, Y)
            g = nan_to_zero(g)

            with torch.no_grad():
                d_step = -alpha * normalize(g)
                delta = delta + d_step
                delta = proj(delta, maxnorm=eps)
                delta = delta.detach()
        delta_sum = delta_sum+delta

    x_adv = X + delta_sum/n_attack_samples
    return x_adv.detach()

x_adv = pgd_attack(inception, x, target_y)
plot_prediction(x_adv)

batch_size = 1

# preprocess transform for ImageNet V2
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

images = ImageNetV2Dataset(transform=preprocess)
loader = torch.utils.data.DataLoader(images, batch_size=batch_size, num_workers=16)

# Iterate over the images of ImageNet V2.
# Use only a small subsample because PGD to create adversarial examples is computationally expensive.
def defense_d0(X,model):
    pred = model(X.to('cuda', dtype=torch.float)) # the laziness....
    pred_class = pred.argmax().cpu().item()
    return pred_class

def defense_d1(X,model,sigma):
    if sigma == 0:
        noise = torch.zeros(size=xB.shape)
    else:
        noise = torch.empty(size=xB.shape).normal_(mean=0, std=sigma)
    x_defended = X.to('cuda', dtype=torch.float) + noise.to('cuda', dtype=torch.float)
    pred = model(x_defended) # the laziness....
    pred_class = pred.argmax().cpu().item()
    return pred_class

def defense_d2(X,model,sigma):
    sigmas = [sigma,sigma/2,sigma/3*2,sigma/3]

    pred = model(X.to('cuda', dtype=torch.float)) # the laziness....
    pred_sum = torch.zeros(size=pred.shape).to('cuda', dtype=torch.float)

    for sigma in sigmas:
        noise = torch.empty(size=X.shape).normal_(mean=0, std=sigma)
        x_defended = X.to('cuda', dtype=torch.float)+noise.to('cuda', dtype=torch.float)
        pred = model(x_defended)
        pred_sum = pred_sum+pred

    mean_pred = pred_sum/(len(sigmas))
    pred_class = mean_pred.argmax().cpu().item()

    return pred_class

def attack_a1(model,X,target_y):
    x_adv = pgd_attack(model, X, target_y)
    return x_adv

def attack_a2(model,X,target_y,sigma,n_attack_samples):

    x_adv= better_pgd_attack(sigma, n_attack_samples, model, X, target_y)

    return x_adv

max_imgs = 200
n_attack_samples = 20
sigmas = [0.0001, .001,.01,.02,.03,0.03759,.04,.05,.06,.07,.09,.11,.13,.15,.17,.19,.21,.25,.35,.5,1.]
#
# max_imgs = 1
# n_attack_samples = 1
# sigmas = [.03,.04,]

num_correct_clean_d0=np.zeros((len(sigmas)))
num_correct_clean_d1=np.zeros((len(sigmas)))
num_correct_clean_d2=np.zeros((len(sigmas)))
num_correct_rob_a1d0=np.zeros((len(sigmas)))
num_correct_rob_a1d1=np.zeros((len(sigmas)))
num_correct_rob_a2d1=np.zeros((len(sigmas)))
num_correct_rob_a2d2=np.zeros((len(sigmas)))
num_adv_success_a1d0=np.zeros((len(sigmas)))
num_adv_success_a1d1=np.zeros((len(sigmas)))
num_adv_success_a2d1=np.zeros((len(sigmas)))
num_adv_success_a2d2=np.zeros((len(sigmas)))

num_imgs = 0
for i, (xB, yB) in enumerate(loader):
    if num_imgs >= max_imgs:
        print(num_imgs)
        break
    else:
        #
        # import pdb
        # pdb.set_trace()

        for s,sigma in enumerate(sigmas):

            if yB == target_y:
                pass  # skip images that are already in the target class
            else:

                # no defense, attack a1
                x_adv = attack_a1(inception, xB, target_y)
                class_adv_defended = defense_d0(x_adv, inception)
                class_clean_defended = defense_d0(xB, inception)

                if class_adv_defended == target_y:
                    num_adv_success_a1d0[s] = num_adv_success_a1d0[s] + 1
                if class_clean_defended == yB:
                    num_correct_clean_d0[s] = num_correct_clean_d0[s] + 1
                if class_adv_defended == yB:
                    num_correct_rob_a1d0[s] = num_correct_rob_a1d0[s] + 1

                # defense d1, attack a1
                x_adv = attack_a1(inception, xB, target_y)
                class_adv_defended = defense_d1(x_adv, inception,sigma=sigma)
                class_clean_defended = defense_d1(xB, inception,sigma=sigma)

                if class_adv_defended == target_y:
                    num_adv_success_a1d1[s] = num_adv_success_a1d1[s] + 1
                if class_clean_defended == yB:
                    num_correct_clean_d1[s] = num_correct_clean_d1[s] + 1
                if class_adv_defended == yB:
                    num_correct_rob_a1d1[s] = num_correct_rob_a1d1[s] + 1

                # defense d1, attack a2
                x_adv = attack_a2(inception, xB, target_y, sigma=sigma, n_attack_samples=n_attack_samples)
                class_adv_defended = defense_d1(x_adv, inception,sigma=sigma)
                class_clean_defended = defense_d1(xB, inception,sigma=sigma)

                if class_adv_defended == target_y:
                    num_adv_success_a2d1[s] = num_adv_success_a2d1[s] + 1
                if class_adv_defended == yB:
                    num_correct_rob_a2d1[s] = num_correct_rob_a2d1[s] + 1

                # defense d2, attack a2
                x_adv = attack_a2(inception, xB, target_y, sigma=sigma, n_attack_samples=n_attack_samples)
                class_adv_defended = defense_d2(x_adv, inception,sigma=sigma)
                class_clean_defended = defense_d2(xB, inception,sigma=sigma)

                if class_adv_defended == target_y:
                    num_adv_success_a2d2[s] = num_adv_success_a2d2[s] + 1
                if class_clean_defended == yB:
                    num_correct_clean_d2[s] = num_correct_clean_d2[s] + 1
                if class_adv_defended == yB:
                    num_correct_rob_a2d2[s] = num_correct_rob_a2d2[s] + 1

        num_imgs += 1
        print(num_imgs)

from plotting import plot_data

plot_data(sigmas=sigmas,
        num_imgs=num_imgs,
        num_adv_success_a1d0=num_adv_success_a1d0,
          num_correct_clean_d0=num_correct_clean_d0,
          num_correct_rob_a1d0=num_correct_rob_a1d0,
        num_adv_success_a1d1=num_adv_success_a1d1,
          num_correct_clean_d1=num_correct_clean_d1,
          num_correct_rob_a1d1=num_correct_rob_a1d1,
        num_adv_success_a2d1=num_adv_success_a2d1,
          num_correct_rob_a2d1=num_correct_rob_a2d1,
        num_adv_success_a2d2=num_adv_success_a2d2,
          num_correct_clean_d2=num_correct_clean_d2,
          num_correct_rob_a2d2=num_correct_rob_a2d2)

#
# print(sigmas)
# print(num_correct_clean)
# print(num_correct_rob)
# print(num_correct_rob2)
# print(num_adv_success)
# print(num_better_adv_success)
import pdb
pdb.set_trace()
        # .......

sigmas_new=np.array(sigmas)
log_sig=np.log(sigmas_new)

plot_data(sigmas=log_sig,
        num_imgs=num_imgs,
        num_adv_success_a1d0=num_adv_success_a1d0,
          num_correct_clean_d0=num_correct_clean_d0,
          num_correct_rob_a1d0=num_correct_rob_a1d0,
        num_adv_success_a1d1=num_adv_success_a1d1,
          num_correct_clean_d1=num_correct_clean_d1,
          num_correct_rob_a1d1=num_correct_rob_a1d1,
        num_adv_success_a2d1=num_adv_success_a2d1,
          num_correct_rob_a2d1=num_correct_rob_a2d1,
        num_adv_success_a2d2=num_adv_success_a2d2,
          num_correct_clean_d2=num_correct_clean_d2,
          num_correct_rob_a2d2=num_correct_rob_a2d2,log=True)