# dogsVScats

来源：[GitHub - gzshan/dogsVScats: 图像二分类问题 猫狗大战 pytorch CNN](https://github.com/gzshan/dogsVScats)

图像二分类问题 猫狗大战 pytorch CNN

使用ResNet34网络模型。

### 使用visdom监测运行过程

```bash
python -m visdom.server
```

然后浏览器打开`localhost:8097`，visdom使用默认的8097端口

<img src="image/README/1642180701224.png" alt="1642180701224.png" style="zoom:50%;" />

利用visdom来记录训练过程中loss的变化和每次epoch的accuracy变化图像。

每`print_freq`一组记录`loss`变化图像：

```python
if ii % opt.print_freq == opt.print_freq - 1:
    vis.plot('loss', loss_meter.value()[0])
```

记录每次训练`accuracy`变化图像：

```python
vis.plot('val_accuracy', val_accuracy)
```

计算验证集上的指标及可视化：

```python
vis.log(
    "epoch: {epoch}, lr:{lr}, loss: {loss}, train_cm: {train_cm}, val_cm: {val_cm}"
    .format(epoch=epoch,
            loss=loss_meter.value()[0],
            val_cm=str(val_cm.value()),
            train_cm=str(confusion_matrix.value()),
            lr=lr))
```

### 训练过程前的初始化

#### 加载网络

```python
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(512, 2)
if opt.use_gpu:  # GPU
    model.cuda()
```

#### 处理数据

加载数据，从给定的训练集中划分出前80%作训练集，后20%做验证集，测试集不做乱序处理。

```python
train_data = DogCat(opt.train_data_root, train=True)  #训练集
val_data = DogCat(opt.train_data_root, train=False)  #验证集，从训练集划分出20%作测试集

train_dataloader = DataLoader(train_data,
                              opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers)
val_dataloader = DataLoader(val_data,
                            opt.batch_size,
                            shuffle=False, # 训练集不做乱序处理
                            num_workers=opt.num_workers)
```

#### 优化器选择

作者选用`SGD`优化器，即随机梯度下降。并且和之前的看过的代码相比，多设置了一个参数，权值衰减`weight_decay`，用于调节模型复杂度对损失函数的影响，防止过拟合。

```python
optimizer = t.optim.SGD(model.parameters(),
                        lr=opt.lr,
                        weight_decay=opt.weight_decay)
```

#### 初始化统计指标

`meter.AverageValueMeter()`：torchnet.meter是用来“跟踪“一些统计量的，也就是能够在一段“历程“中记录下某个统计量在迭代过程中不断变化的值，并统计相关的量。

`meter.ConfusionMeter(2)`：混淆矩阵，每一列表达了分类器对于样本的类别预测，每一行表达了真实类别。所以正确分类都在矩阵的对角线上，**能够很容易的看到有没有将样本的类别给混淆了**。

<img src="https://s2.loli.net/2022/01/15/FHO341SMnCfUsip.png" alt="image-20220115224947919" style="zoom:50%;" />

```python
loss_meter = meter.AverageValueMeter() # 跟踪loss统计量
confusion_matrix = meter.ConfusionMeter(2) # 混淆矩阵
previous_loss = 1e10 # 初始化loss
```

#### 下降学习率

作者在训练过程中，当出现loss不再下降的时候，下降学习率。

初始学习率为`lr=0.001`，学习率的降低速率`lr_decay=0.95`。

```python
"""如果损失不再下降，则降低学习率"""
if loss_meter.value()[0] > previous_loss:
    lr = lr * opt.lr_decay
    for param_group in optimizer.param_groups: # 下降SGD优化器中的学习率
        param_group["lr"] = lr
```



### 开始训练

在每一轮训练前都要对统计指标初始化：

```python
loss_meter.reset()
confusion_matrix.reset()
```

为了让训练进度可视化，构建tqdm进度条`processBar = tqdm(train_dataloader, *unit*='step')`。

然后遍历`processBar`（也就是训练集train_dataloader）

```python
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
    confusion_matrix.add(out.detach(), target.detach())
```

每轮训练都保留下当前的模型

```python
name = time.strftime('model' + '%m%d_%H_%M_%S.pth')
t.save(model.state_dict(), 'checkpoints/' + name)  # 保存epoch个模型
```

### 计算验证集上的指标

```python
val_cm, val_accuracy = val(model, val_dataloader)
processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % (epoch, opt.max_epoch, loss.item(), val_accuracy.item()))
print("epoch:", epoch, "loss:", loss_meter.value()[0], "accuracy:", val_accuracy)
```

验证模式(计算混淆矩阵和正确率)：

```python
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
```

### 测试部分

数据集加载

```python
test_data = DogCat(opt.test_data_root, test=True)
test_dataloader = DataLoader(test_data,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=opt.num_workers)
```

模型初始化：

```python
model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 2) # 设置全连接层
    model.load_state_dict(t.load('./model.pth')) # 加载模型
    if opt.use_gpu: model.cuda() # gpu优化
    model.eval() # 将模型设置为验证模式
```

将预测结果以`(id, res)` 的结果返回到results list中，如`(5, "Dog")`表示`5.jpg`是狗。

```python
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
```

最后将结果写到`csv`文件中。

