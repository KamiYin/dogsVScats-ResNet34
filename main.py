"""
主程序:主要完成四个功能
（1）训练:定义网络，损失函数，优化器，进行训练，生成模型
（2）验证:验证模型准确率
（3）测试:测试模型在测试集上的准确率
（4）help:打印log信息
"""

import torch
from config import opt
import os
# import models
import torch as t
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from torch.autograd import Variable
from torchvision import models
from torch import nn
import time
import csv
from tqdm import tqdm

"""模型训练:定义网络，定义数据，定义损失函数和优化器，训练并计算指标，计算在验证集上的准确率"""
def train(**kwargs):
    """根据命令行参数更新配置"""
    opt.parse(kwargs)
    vis = Visualizer(opt.env) # visdom环境

    """(1)step1:加载网络，若有预训练模型也加载"""
    # model = getattr(models, opt.model)() # getattr()用于返回models对象的opt.model()属性值
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2) # 设置全连接层的张量大小
    # if opt.load_model_path:
    #     model.load(opt.load_model_path)
    if opt.use_gpu: model.cuda() # GPU

    """(2)step2:处理数据"""
    train_data = DogCat(opt.train_data_root, train=True)  #训练集
    val_data = DogCat(opt.train_data_root, train=False)  #验证集，从训练集划分出20%作测试集

    train_dataloader = DataLoader(train_data,
                                  opt.batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,
                                opt.batch_size,
                                shuffle=False, # 测试集不做乱序处理
                                num_workers=opt.num_workers)

    """(3)step3:定义损失函数和优化器"""
    criterion = t.nn.CrossEntropyLoss()  # 定义loss计算方法，交叉熵损失
    lr = opt.lr  #学习率
    # 使用SGD优化器
    optimizer = t.optim.SGD(model.parameters(),
                            lr=opt.lr,
                            weight_decay=opt.weight_decay)

    """(4)step4:统计指标，平滑处理之后的损失，还有混淆矩阵"""
    # torchnet.meter 提供了一种标准化的方法来测量一系列不同的测量，这使得测量模型的各种属性变得容易。
    loss_meter = meter.AverageValueMeter() # 跟踪loss统计量
    confusion_matrix = meter.ConfusionMeter(2) # 混淆矩阵
    previous_loss = 1e10 # 初始化loss

    """(5)开始训练"""
    for epoch in range(opt.max_epoch):
        # 初始化loss的统计量和混淆矩阵
        loss_meter.reset()
        confusion_matrix.reset()
        processBar = tqdm(train_dataloader, unit='step') # 构建tqdm进度条

        for ii, (data, label) in enumerate(processBar):
            # 训练模型参数
            data_in = Variable(data)
            target = Variable(label) # 将数据放置在PyTorch的Variable节点中
            if opt.use_gpu:
                data_in = data_in.cuda()
                target = target.cuda()

            optimizer.zero_grad() # 梯度清零
            out = model(data_in) # 调用forward()方法计算网络输出值

            loss = criterion(out, target) # 交叉熵损失
            loss.backward()  # 反向传播

            optimizer.step() # 使用SGD优化器更新参数

            # 更新统计指标及可视化
            loss_meter.add(loss.item())
            # print(out.shape, target.shape)
            confusion_matrix.add(out.detach(), target.detach())

            # 每print_freq组数据记录一次loss（平滑处理），在visdom输出
            if ii % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()
            processBar.set_description("[%d/%d] Loss: %.4f" % (epoch, opt.max_epoch, loss.item()))

        #model.save()
        name = time.strftime('model' + '%m%d_%H_%M_%S.pth')
        t.save(model.state_dict(), 'checkpoints/' + name)  # 保存epoch个模型

        """计算验证集上的指标及可视化"""
        val_cm, val_accuracy = val(model, val_dataloader)
        vis.plot('val_accuracy', val_accuracy)
        vis.log(
            "epoch: {epoch}, lr:{lr}, loss: {loss}, train_cm: {train_cm}, val_cm: {val_cm}"
            .format(epoch=epoch,
                    loss=loss_meter.value()[0],
                    val_cm=str(val_cm.value()),
                    train_cm=str(confusion_matrix.value()),
                    lr=lr))

        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % (epoch, opt.max_epoch, loss.item(), val_accuracy.item()))
        print("epoch:", epoch, "loss:", loss_meter.value()[0], "accuracy:", val_accuracy)
              
        """如果损失不再下降，则降低学习率"""
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            for param_group in optimizer.param_groups: # 下降SGD优化器中的学习率
                param_group["lr"] = lr

        previous_loss = loss_meter.value()[0]
    


"""计算模型在验证集上的准确率等信息"""
@t.no_grad()
def val(model, dataloader): # 返回混淆矩阵和正确率
    model.eval()  # 将模型设置为验证模式
    confusion_matrix = meter.ConfusionMeter(2) # 混淆矩阵
    for ii, data in enumerate(dataloader):
        data_in, label = data
        with torch.no_grad: # 不自动求导
            val_input = Variable(data_in)
            val_label = Variable(label.long())
        if opt.use_gpu: # GPU优化
            val_input = val_input.cuda()
            val_label = val_label.cuda()

        out = model(val_input)
        confusion_matrix.add(out.detach().squeeze(), label.long())

    model.train()  #模型恢复为训练模式
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

    return confusion_matrix, accuracy


"""测试部分，使用模型对测试集数据进行分类，并将分类结果存储到csv文件中"""
def test(**kwargs):
    opt.parse(kwargs)

    # data
    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers)

    #model
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2) # 设置全连接层
    model.load_state_dict(t.load('./model.pth')) # 加载模型
    if opt.use_gpu: model.cuda() # gpu优化
    model.eval() # 将模型设置为验证模式

    results = []
    for ii, (data, label) in enumerate(test_dataloader):
        data_in = Variable(data, volatile=True)
        if opt.use_gpu: data_in = data_in.cuda()
        out = model(data_in)
        label = label.numpy().tolist() # 图片id
        _, predicted = t.max(out.data, 1)
        predicted = predicted.data.cpu().numpy().tolist()
        res = ""
        for (i, j) in zip(label, predicted):
            res = "Dog" if j == 1 else "Cat"
            results.append([i, "".join(res)])

    write_csv(results, opt.result_file)
    return results


"""输出识别结果到csv文件"""
def write_csv(results, file_name):
    with open(file_name, "w") as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


if __name__ == '__main__':
    # import fire
    # fire.Fire()
    test()