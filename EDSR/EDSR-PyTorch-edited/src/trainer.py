import os
import math
from decimal import Decimal

import utility
import numpy as np
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from tools import convert_visuals_to_numpy

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        
    def psnr(self, img1, img2):
        mse_value = np.mean((img1 - img2)**2)

        return 20. * np.log10(255. / np.sqrt(mse_value))

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        # self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        psnrs = []
        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            mean_psnr = 0
            for idx_scale, scale in enumerate(self.scale):
                # d.dataset.set_scale(idx_scale)
                # print(len(d), d[0].size())
                c = [(d[0][i].unsqueeze(0), d[1][i].unsqueeze(0), d[2][i]) for i in range(d[0].size(0))]
                for lr, hr, filename in tqdm(c, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)



                    # original quantize used in the paper
                    # sr = utility.quantize(sr, self.args.rgb_range)

                    # if idx_data == 1:
                    #     print(sr)
                    
                    # change back to 0-255
                    sr = convert_visuals_to_numpy(sr)
                    lr = convert_visuals_to_numpy(lr)
                    hr = convert_visuals_to_numpy(hr)
                    
                    

                    if not self.args.test_only:
                        tem_psnr = self.psnr(hr, sr)
                        self.ckp.log[-1, idx_data, idx_scale] += tem_psnr
                        mean_psnr += tem_psnr
                        # utility.calc_psnr(
                        #     sr, hr, scale, self.args.rgb_range, dataset=False # changed
                        # )
                        
                    # if not self.args.test_only:
                        

                    save_list = [sr]
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)
                if not self.args.test_only:
                    self.ckp.log[-1, idx_data, idx_scale] /= len(c)
                    mean_psnr /= len(c)
                    psnrs.append(mean_psnr)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            "FFHQ",
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale] + 1
                        )
                    )
        if not self.args.test_only:
            final_psnr = sum(psnrs)/len(psnrs)
            print(f"Test psnr: {final_psnr}")
            self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
            self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

