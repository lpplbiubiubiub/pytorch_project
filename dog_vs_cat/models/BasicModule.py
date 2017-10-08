# coding:utf8
import torch as t
import time

class BasicModule(t.nn.Module):
    def __init__(self, opt=None):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, model_path):
        """
        load model according name
        """
        model_dict = self.state_dict()
        pretrained_dict = t.load(model_path)
        # 将pretrained_dict里不属于model_dict的键剔除掉
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        self.load_state_dict(model_dict)

    def save(self, name=None):
        """
        save model
        """
        if name is None:
           prefix = 'checkpoints/' + self.model_name + '_'
           name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name
