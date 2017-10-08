#coding:utf8
import warnings
class DefaultConfig(object):
   env = 'default' # visdom 环境
   model = 'EasyNet' # 使用的模型，名字必须与models/__init__.py中的名字一致

   train_data_root = '/home/xksj/Data/lp/dog_vs_cat' # 训练集存放路径
   load_model_path = 'checkpoints/model.pth' # 加载预训练的模型的路径，为None代表不加载

   batch_size = 128 # batch size
   use_gpu = True # use GPU or not
   num_workers = 4 # how many workers for loading data
   print_freq = 20 # print info every N batch

   debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
   result_file = 'result.csv'

   max_epoch = 10
   lr = 0.01 # initial learning rate
   lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
   weight_decay = 1e-4 # 损失函数
def parse(self, kwargs):
     '''
     根据字典kwargs 更新 config参数
     '''
     # 更新配置参数
     for k, v in kwargs.iteritems():
         if not hasattr(self, k):
             # 警告还是报错，取决于你个人的喜好
             warnings.warn("Warning: opt has not attribut %s" %k)
         setattr(self, k, v)

     # 打印配置信息
     print('user config:')
     # self.__class__等价于调取类
     # __dict__ 获取属性
     # 在这里self指的是DefaultConfig的类属性
     for k, v in self.__class__.__dict__.iteritems():
         # 这里 __开头的是class自带的属性 要显示自定义属性
         if not k.startswith('__'):
             print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()
