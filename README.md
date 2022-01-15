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
    "epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}"
    .format(epoch=epoch,
            loss=loss_meter.value()[0],
            val_cm=str(val_cm.value()),
            train_cm=str(confusion_matrix.value()),
            lr=lr))
```

### 训练过程

#### 

#### 加载网络

```python
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(512, 2)
if opt.use_gpu:  # GPU
    model.cuda()
```

#### 

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

作者选用`SGD`优化器，即随机梯度下降。

#### 初始化统计指标

`meter.AverageValueMeter()`：torchnet.meter是用来“跟踪“一些统计量的，也就是能够在一段“历程“中记录下某个统计量在迭代过程中不断变化的值，并統计相关的量。

`meter.ConfusionMeter(2)`：混淆矩阵，每一列表达了分类器对于样本的类别预测，每一行表达了真实类别。所以正确分类都在矩阵的对角线上，能够很容易的看到有没有将样本的类别给混淆了。

<img src="https://s2.loli.net/2022/01/15/FHO341SMnCfUsip.png" alt="image-20220115224947919" style="zoom:50%;" />

```python
loss_meter = meter.AverageValueMeter() # 跟踪loss统计量
confusion_matrix = meter.ConfusionMeter(2) # 混淆矩阵
previous_loss = 1e10 # 初始化loss
```



#### 下降学习率

作者在训练过程中，当出现loss不再下降的时候，下降学习率。

初始学习率为`lr=0.001`，学习率的降低速率`lr_decay=0.95`。

