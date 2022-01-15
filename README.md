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

加载数据，从给定的训练集中划分出前80%作训练集，后20%做验证集。

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

#### 下降学习率

作者在训练过程中，当出现loss不再下降的时候，下降学习率。

初始学习率为`lr=0.001`，学习率的降低速率`lr_decay=0.95`。

