import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import torch.nn as nn

import hi_data
from hi_options.config_hifacegan import TrainOptions, TestOptions

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    train_opt = TrainOptions()
    test_opt = TestOptions()
    if args.test_only:
        # test_opt.dataroot = "..\\..\\data\\FFHQ\\test\\LQ"
        test_opt.dataroot = "..\\..\\data\\FFHQ\\val\\LQ"
        test_opt.dataset_mode = "truetest"
    
    class InterLoader(object):
        loader_train = hi_data.create_dataloader(train_opt)
        loader_test = hi_data.create_dataloader(test_opt)
    
    dataloader = InterLoader()
    
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            # loader = data.Data(args)
            
            _model = model.Model(args, checkpoint)
            _model.sub_mean = nn.Identity()
            _model.add_mean = nn.Identity()
            # print(_model)
            print(sum(p.numel() for p in _model.parameters()))
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, dataloader, _model, _loss, checkpoint)
            if args.test_only:
                t.test()
            else:
                while not t.terminate():
                    # if not args.test_only:
                    t.train()
                    t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
