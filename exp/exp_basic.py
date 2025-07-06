import os
import mindspore as ms
import mindspore.context as context
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=self.args.gpu)
            device = 'GPU'
            print('Use GPU: GPU:{}'.format(self.args.gpu))
        else:
            context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
            device = 'CPU'
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    