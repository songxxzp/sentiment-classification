### Run
```
# 下载数据：
git lfs install
git lfs pull

# 训练一个模型(CNN, RNN, MLP)：
python main.py

# CNN, RNN, MLP实验：
python experiment.py

# bert：
python bert.py

# 统计模型：
shallow_machine_learning.py
```

### 目录
```
./log: bert TensorBoard(create on run)
./runs: CNN, RNN, MLP TensorBoard(create on run)
./results: CNN, RNN, MLP results in json
./tensorboards: tensorboards for homework
```

### TensorBoard
```
cd ./runs
```
#### regex example
##### 过滤所有仅保存`hparams`的`board`
```
windows:
^[^\\]*(\\[^\\]*){8}$
linux:
^[^/]*(/[^/]*){8}$
```
|    | `^[^\\]*(\\[^\\]*){8}$` |
|----|----|
| 保留 | `20230507094810\TextCNN\094813\SGD\max_pool1d\dropout=0\lr=1e-05\early_stop_epoch=0\froze_embedding=True` |
| 过滤 | `20230507094810\TextCNN\094813\SGD\max_pool1d\dropout=0\lr=1e-05\early_stop_epoch=0\froze_embedding=True\1683424128.4343944` |

##### 过滤单一条件
```
.*TextCNN*
.*AdamW*

.*lr\=1e-5*
.*lr\=0\.0001*
```

##### 多个条件组合
```
.*TextCNN*
.*AdamW*
.*lr\=1e-5*
^[^\\]*(\\[^\\]*){8}$

windows:
(?=.*TextCNN*)(?=.*AdamW*)(?=.*lr\=1e-5*)^[^\\]*(\\[^\\]*){8}$
linux:
(?=.*TextCNN*)(?=.*AdamW*)(?=.*lr\=1e-5*)^[^/]*(/[^/]*){8}$
```
|    | `(?=.*TextCNN*)(?=.*AdamW*)^[^\\]*(\\[^\\]*){8}$` |
|----|----|
| 保留 | `20230507094810\TextCNN\094813\AdamW\max_pool1d\dropout=0\lr=1e-05\early_stop_epoch=0\froze_embedding=True` |
| 过滤 | `20230507094810\TextCNN\094813\AdamW\max_pool1d\dropout=0\lr=1e-05\early_stop_epoch=0\froze_embedding=True\1683424128.4343944` |
| 过滤 | `20230507094810\TextCNN\094813\SGD\max_pool1d\dropout=0\lr=1e-05\early_stop_epoch=0\froze_embedding=True` |
