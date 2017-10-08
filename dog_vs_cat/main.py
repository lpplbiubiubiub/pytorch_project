#coding:utf8
from config import opt
import os
import torch as t
import models
from data.dataset import DogVsCatDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from utils.visualize import Visualizer

def test(**kwargs):
    opt.parse(kwargs)
    # ipdb调试相关
    import ipdb;
    ipdb.set_trace()
    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # data
    train_data = DogVsCatDataset(opt.test_data_root,test=True)
    test_dataloader = DataLoader(train_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers)
    results = []
    for ii,(data,path) in enumerate(test_dataloader):
        input = t.autograd.Variable(data,volatile = True)
        if opt.use_gpu: input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score)[:,0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()

        batch_results = [(path_,probability_) for path_,probability_ in zip(path,probability) ]

        results += batch_results
    write_csv(results,opt.result_file)

    return results

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda()

    # step2: data
    train_data = DogVsCatDataset(opt.train_data_root,is_train=True)
    val_data = DogVsCatDataset(opt.train_data_root,is_train=False)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),lr = lr,weight_decay = opt.weight_decay)

    # step4: meters
    # loss_meter = meter.AverageValueMeter()
    # confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    # train
    for epoch in range(opt.max_epoch):

        # loss_meter.reset()
        # confusion_matrix.reset()
        print("training...")
        for ii,(data,label) in enumerate(train_dataloader):

            # train model
            input = Variable(data)
            target = Variable(label)
            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()


            # meters update and visualize
            # loss_meter.add(loss.data[0])
            # confusion_matrix.add(score.data, target.data)

            if ii%opt.print_freq==opt.print_freq-1:
                vis.plot('loss', loss.data[0])

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb;
                    ipdb.set_trace()


        model.save()

        # validate and visualize
        val_cm,val_accuracy = val(model,val_dataloader)
        vis.plot('val_accuracy',val_accuracy)

        # 如果loss不再下降 则降低学习率
        """
        if loss.data[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss.data[0]
        """
def val(model,dataloader):
    '''
    计算模型在验证集上的准确率等信息
    '''
    model.eval()
    match_num = 0.
    match_time = 0
    for ii, data in enumerate(dataloader):

        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.type(t.LongTensor), volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
            val_label = val_label.cuda()
        score = model(val_input)

        pred = score.cpu().data.max(1)[1] # get the index of the max log-probability
        match_num += pred.eq(val_label.cpu().data).cpu().sum()


    model.train()

    accuracy = 100. * (match_num) / len(dataloader.dataset)
    return None, accuracy

def help():
    '''
    打印帮助的信息： python file.py help
    '''

    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example:
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:'''.format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()
