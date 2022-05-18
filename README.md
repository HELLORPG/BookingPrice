# README

```wiki
@author HELLORPG
@date 2022.5.11
@description 2022年，高级机器学习，第二次作业。
```

## TODO

使用给定的 [数据集和描述](https://cs.nju.edu.cn/liyf/aml22/assignment2.htm#1) ，根据房屋提供的各类信息（例如卫生间情况/地理位置），给出房屋租金价格等级（From 0 to 5）的预测。


## Dataset

数据包括 [训练集](./dataset/train.csv) 和 [测试集](./dataset/test.csv) 两部分，由于是课程作业，因此测试集部分的标签真值结果并未公布。

每一条数据应该包括 16 个属性以及 1 一个标签（测试集没有标签），其中 16 个属性的说明如下：
- description：房屋的信息描述，粗略查看具有英语（主要）/简中/繁中三种语言，并且会出现 Emoji 表情。`Example: 🚘 FREE CAR SPACE AVAILABLE ON THE BUILDING😊<br /><br />Relax and enjoy your stay at this gorgeous, sun-filled beachside apartment,located only moments away from the sand and surf of Bondi Beach. `
- neighbourhood：表示房屋的位置信息。`Example: Waverley`
- latitude：房子的纬度信息。`Example: -33.88882`
- longitude：房子的精度信息。`Example: 151.27456`
- type：房屋的种类，例如"整套房源"或是"独立房间"。`Example: Entire home/apt`
- accommodates：房源可以容纳多少人。`Example: 2`
- bathrooms：卫生间信息，独立或是公用，以及个数。`Example: 1 private bath`
- bedrooms：卧室的个数。`Example: 1.0`
- amenities：有哪些设施。`Example: ["Gym", "Bed linens", "Shampoo", "Coffee maker", "Hair dryer", "TV", "Heating", "Washer", "Iron", "Free parking on premises", "Smoke alarm", "Essentials", "Cooking basics", "Private entrance", "Shower gel", "Extra pillows and blankets", "Kitchen", "Oven", "Carbon monoxide alarm", "Long term stays allowed", "Microwave", "Hot water", "Air conditioning", "Dryer", "Dedicated workspace", "Hangers", "Wifi", "Refrigerator"]`
- reviews：评论数量。`Example: 42`
- review_rating：评价的平均分数。`Example: 96.0`
- review_scores_A：对A项目的平均评分。`Example: 10.0`
- review_scores_B
- review_scores_C
- review_scores_D
- instant_bookable：房源是否可以立即预定，使用 t 或者 f 表示。`Example: f`

数据集的原始描述文件：[Description of the Dataset](./dataset/README.md)。



## Run

本项目中可运行的所有模块都集成在了`run.py`中，可以使用`python run.py [--args-name args-value]`的形式运行。

整理了目前支持的运行模块以及各个参数的设置方法：

### 划分数据集
```bash
python run.py --part split-train-val --input-path <original train set path> --train-path <train set path> --val-path <val set path> [--train-ratio <train set ratio> / --train-size <train set size> / --val-size <val set size>] 
```
末尾`[]`中的三个参数选择一个填入即可，如果重复写入则会按照上述给出的顺序优先采用靠前的参数。

Example:
```bash
python run.py --part split-train-val --input-path ./dataset/train.csv --train-path ./dataset/split/train.csv --val-path ./dataset/split/val.csv --val-size 5000
```

### 可视化房屋经纬度（位置）和价格的关系
```bash
python run.py --part location-visualization --input-path <data set path> --output-path [figure save path]
```
只需要指定运行模块和输入输出路径即可，如果不提供输出路径，则仅仅只会展示在屏幕上，而不保存；提供了输出路径，则不会展示，而仅保存。

Example:
```bash
python run.py --part location-visualization --input-path ./dataset/split/train.csv --output-path ./outputs/location_visualization/train.png
```


