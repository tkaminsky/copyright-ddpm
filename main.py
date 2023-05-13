import copy
import json
import os
import warnings
from absl import app, flags

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange
from torch.utils.data import Dataset

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from copyright import cp_k_model
from model import UNet
from score.both import get_inception_and_fid_score


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 80000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')

# Pretrained checkpoint for training
flags.DEFINE_string('ckpt', './logs/DDPM_CIFAR10_EPS/', help='checkpoint dir')

flags.DEFINE_bool('is_copyright_safe', False, help='Whether using the copyright-safe model for evaluation')

flags.DEFINE_string('dataset', 'full', help='copy, safe, or full––which parts of the dataset to use')

device = torch.device('cuda:0')

class CopyDataset(Dataset):
    def __init__(self, X, label):
        self.data = X
        self.label = label

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx] # -> a batch 

    def __len__(self):
        return len(self.data)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images

def evaluate_copyright_safe(model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc=desc):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            # x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = model.sample().cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train():

    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    # Split the dataset in half
    dataset1 = torch.utils.data.Subset(dataset, range(len(dataset) // 2))
    dataset2 = torch.utils.data.Subset(dataset, range(len(dataset) // 2, len(dataset)))
    #dataset1, dataset2 = torch.utils.data.random_split(dataset, [25000, 25000])
    #ds = torch.utils.data.ConcatDataset([dataset1, dataset2])
    
    s1 = dataset1[7]
    
    if FLAGS.dataset == 'copy':
        # Copy the first sample
        # Save the image from the first sample to current directory
        save_image(s1[0], 'copy_image_copy.png')

        ds1_len = len(dataset1)

        num_copies = ds1_len // 10

        s1_tensor = s1[0]
        print(s1_tensor.shape)
        # Copies of s1_tensor of dim (num_copies, 3, 32, 32)
        s1_tcpies = s1_tensor.repeat(num_copies, 1, 1, 1)
        s1_labels = torch.tensor(s1[1])
        s1_lcpies = s1_labels.repeat(num_copies, 1)
        labels = [s1[1] for i in range(num_copies)]
        print(s1_lcpies.shape)
        ds_cpy = torch.utils.data.TensorDataset(s1_tcpies, s1_lcpies)
        ds_cpy = CopyDataset(s1_tcpies, labels)

        dataset = torch.utils.data.ConcatDataset([dataset1, ds_cpy])
    elif FLAGS.dataset == 'safe':
        # Save the image from the first sample to current directory
        save_image(s1[0], 'copy_image_safe.png')
        dataset = dataset2
    else:
        # Save the image from the first sample to current directory
        save_image(s1[0], 'copy_image_full.png')
        ds1_len = len(dataset1)

        num_copies = ds1_len // 10

        s1_tensor = s1[0]
        print(s1_tensor.shape)
        # Copies of s1_tensor of dim (num_copies, 3, 32, 32)
        s1_tcpies = s1_tensor.repeat(num_copies, 1, 1, 1)
        s1_labels = torch.tensor(s1[1])
        s1_lcpies = s1_labels.repeat(num_copies, 1)
        labels = [s1[1] for i in range(num_copies)]
        print(s1_lcpies.shape)
        ds_cpy = torch.utils.data.TensorDataset(s1_tcpies, s1_lcpies)
        ds_cpy = CopyDataset(s1_tcpies, labels)

        dataset = torch.utils.data.ConcatDataset([dataset, ds_cpy])
    # Otherwise, use the full dataset
    print(f"Using dataset {FLAGS.dataset}")
    # ADDED CODE END


    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    ema_model = copy.deepcopy(net_model)

    ckpt = torch.load(os.path.join(FLAGS.ckpt, 'ckpt.pt'))
    net_model.load_state_dict(ckpt['net_model'])
    ema_model.load_state_dict(ckpt['ema_model'])

    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2
    writer = SummaryWriter(FLAGS.logdir)
    writer.add_image('real_sample', grid)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    
    FLAGS.total_steps = 50000

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.3f' % loss)

            # sample
            if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                    grid = (make_grid(x_0) + 1) / 2
                    path = os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % step)
                    save_image(grid, path)
                    writer.add_image('sample', grid, step)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt.pt'))

            # evaluate
            if FLAGS.eval_step > 0 and step % FLAGS.eval_step == 0:
                net_IS, net_FID, _ = evaluate(net_sampler, net_model)
                ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def eval():
    if FLAGS.is_copyright_safe:
        model1 = UNet(
            T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
            num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
        sampler = GaussianDiffusionSampler(
            model1, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
        if FLAGS.parallel:
            sampler = torch.nn.DataParallel(sampler)
        
        print("Model 1 loaded")

        model2 = UNet(
            T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
            num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
        sampler2 = GaussianDiffusionSampler(
            model2, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
        if FLAGS.parallel:
            sampler = torch.nn.DataParallel(sampler)
            
        print("Model 2 loaded")
        
        model3 = UNet(
            T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
            num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
        sampler3 = GaussianDiffusionSampler(
            model3, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
        if FLAGS.parallel:
            sampler = torch.nn.DataParallel(sampler)
        
        print("Model 3 loaded")
        
        ckpt1 = torch.load(os.path.join('logs/CIFAR10_full', 'ckpt.pt'))
        model1.load_state_dict(ckpt1['net_model'])
        
        print("Checkpoint 1 loaded")

        ckpt2 = torch.load(os.path.join('logs/CIFAR10_copy', 'ckpt.pt'))
        model2.load_state_dict(ckpt2['net_model'])
        
        print("Checkpoint 2 loaded")

        ckpt3 = torch.load(os.path.join('logs/CIFAR10_safe', 'ckpt.pt'))
        model3.load_state_dict(ckpt3['net_model'])
        
        print("Checkpoint 3 loaded")
        
        FLAGS.batch_size = 128
        FLAGS.num_images = 256
        
        sampler1 = GaussianDiffusionSampler(
            model1, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
        if FLAGS.parallel:
            sampler1 = torch.nn.DataParallel(sampler1)
            
        sampler2 = GaussianDiffusionSampler(
            model2, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
        if FLAGS.parallel:
            sampler2 = torch.nn.DataParallel(sampler2)
        
        sampler3 = GaussianDiffusionSampler(
            model3, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
        if FLAGS.parallel:
            sampler1 = torch.nn.DataParallel(sampler3)

        shape = [FLAGS.batch_size, 3, FLAGS.img_size, FLAGS.img_size]
        model = cp_k_model(sampler1, (sampler2, sampler3), shape, k=500)
        
        print("Final model loaded. Beginning evaluation:")
        
        (IS, IS_std), FID, samples = evaluate_copyright_safe(model)
        print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(
            torch.tensor(samples[:256]),
            os.path.join(FLAGS.logdir, 'samples.png'),
            nrow=16)
    else:
        # model setup
        model = UNet(
            T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
            num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
        sampler = GaussianDiffusionSampler(
            model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
            mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
        if FLAGS.parallel:
            sampler = torch.nn.DataParallel(sampler)

        # load model and evaluate
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt.pt'))
        model.load_state_dict(ckpt['net_model'])
        
        FLAGS.batch_size = 128
        FLAGS.num_images = 256
        
        (IS, IS_std), FID, samples = evaluate(sampler, model)
        print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(
            torch.tensor(samples[:256]),
            os.path.join(FLAGS.logdir, 'samples.png'),
            nrow=16)

        # model.load_state_dict(ckpt['ema_model'])
        # (IS, IS_std), FID, samples = evaluate(sampler, model)
        # print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        # save_image(
        #     torch.tensor(samples[:256]),
        #     os.path.join(FLAGS.logdir, 'samples_ema.png'),
        #     nrow=16)


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
