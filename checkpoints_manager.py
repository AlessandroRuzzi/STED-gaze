"""Copyright 2020 ETH Zurich, Yufeng Zheng, Seonwook Park
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import os
import torch

ckpt_extension = '.pth.tar'
ckpt_fmtstring = 'at_step_%07d' + ckpt_extension


def step_number_from_fname(fpath):
    fname = fpath.split('/')[-1]
    stem = fname.split('.')[0]
    return int(stem.split('_')[-1])


class CheckpointsManager(object):

    def __init__(self, network, output_dir,device):
        self.network = network
        self.output_dir = os.path.realpath(output_dir + '/checkpoints')
        self.device = device

    @property
    def all_available_checkpoint_files(self):
        if not os.path.isdir(self.output_dir):
            return []
        fpaths = [
            (step_number_from_fname(p), self.output_dir + '/' + p)
            for p in os.listdir(self.output_dir)
            if os.path.isfile(self.output_dir + '/' + p)
            and p.endswith(ckpt_extension)
        ]
        fpaths = sorted(fpaths)  # sort by step number
        return fpaths

    def load_last_checkpoint(self,xgaze = False):
        available_fpaths = self.all_available_checkpoint_files
        if len(available_fpaths) > 0:
            step_number, fpath = available_fpaths[-1]
            logging.info('Found weights file: %s' % fpath)
            loaded_step_number = self.load_checkpoint(step_number, fpath, xgaze)
            return loaded_step_number
        return 0

    def load_checkpoint(self, step_number, checkpoint_fpath, xgaze = False):
        assert os.path.isfile(checkpoint_fpath)
        weights = torch.load(checkpoint_fpath,map_location=torch.device(self.device))
       
        print(step_number,checkpoint_fpath)
        # If was stored using DataParallel but being read on 1 GPU
        if torch.cuda.device_count() == 1:
            if next(iter(weights.keys())).startswith('module.'):
                weights = dict([(k[7:], v) for k, v in weights.items()])
        if xgaze:
            self.network.load_state_dict(weights["model_state"])
        else:
            self.network.load_state_dict(weights)
        
        logging.info('Loaded known model weights at step %d' % step_number)
        return step_number

    def save_checkpoint(self, step_number):
        assert os.path.isdir(os.path.abspath(self.output_dir + '/../'))
        fname = ckpt_fmtstring % step_number
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        ofpath = '%s/%s' % (self.output_dir, fname)
        torch.save(self.network.state_dict(), ofpath)
        torch.cuda.empty_cache()
