# group-task4-Hyperspectral

利用覆盖 **400–1000 nm** 的高光谱数据（每像素 **32 个波段**）进行精细地物分类：给定场景中若干像素的谱向量（`band_1 ... band_32`），预测其土地覆盖类别（`label`）。

本仓库实现了一条偏“工程落地”的训练流水线：  
- **大规模 CSV → NumPy memmap**（可流式处理，避免爆内存）  
- **Conv + Transformer 自编码器（AE）**学习光谱表征（可扩展到半监督）  
- **AE + MLP 分类头**进行像素级分类（支持类别不平衡加权）  
- 输出 `predictions.csv`（`id,pred_label`）

---

## 任务与数据

- 分辨率：10 m  
- 光谱范围：400–1000 nm  
- 特征维度：32 bands  
- 评价指标：像素级 Overall Accuracy（OA）

### 文件格式
- `train.csv`
  - `id`：样本编号
  - `band_1 ... band_32`：32个波段反射率
  - `label`：地物类别编号（通常为 0–5）
- `test.csv`
  - `id`
  - `band_1 ... band_32`

> ⚠️ 注意：本仓库代码里 `num_classes` 默认写成 **5**，请根据你的实际标签范围（例如 0–5 为 6 类）改成正确的类别数。


## 仓库结构

```

.
├── custom_dataset.py          # Dataset/DataLoader（支持 .npy memmap）
├── custom_model.py            # ConvTransformerAE + 分类器 + AEClassifier封装
├── custom_trainer.py          # AE训练 & 分类训练（TensorBoard日志、权重保存）
├── save_model.py              # checkpoint 保存/加载（encoder+classifier+opt）
├── data_utils.ipynb           # CSV→memmap、流式统计mean/std、归一化、划分val
├── trainer.ipynb              # 训练自编码器（AE）
├── classifier_train.ipynb     # 训练分类头（可冻结/半冻结AE）
└── eval.ipynb                 # 推理并导出 predictions.csv

```

建议你在项目根目录准备一个 `resources/` 目录放数据与中间产物：

```

resources/
├── train.csv
├── test.csv
├── features.npy / features_norm.npy
├── labels.npy
├── test_features.npy / test_features_norm.npy
└── models/

````

---

## 环境依赖

核心依赖：
- Python 3.12+
- PyTorch 2.8.0 (with CUDA 12.8)
- NumPy
- tqdm
- tensorboard

---

## 模型简介（`custom_model.py`）

### 1) ConvTransformerAE（自编码器）

* **Conv1D stem**：在光谱维上抽局部模式（相邻波段相关、斜率/吸收特征）
* **Transformer Encoder**：建模跨波段的全局依赖关系
* **Latent**：对序列做 mean pooling 后投影到低维 latent
* **Decoder(MLP)**：从 latent 重建原 32 维光谱（MSE）

> 直觉：重建任务给了“自监督信号”，在标注少/类别不均衡时能当正则，还天然适配半监督（无标签样本也能用重建loss训练编码器）。

### 2) AEClassifier（编码器 + 分类器）

* 复用 AE 的 `encode()` 输出 latent
* MLP 分类头输出 `num_classes` logits
* 训练时联合 loss：

  * `CrossEntropy(logits, y)`（可加权）
  * `0.2 * MSE(x_hat, x)`（重建正则项）

---

## 快速开始（推荐按 notebook 顺序）

> 先创建目录（避免保存模型时报错）：

```bash
mkdir -p resources/models logs logs/classifier
```

### Step 0：CSV → NumPy memmap + 归一化（`data_utils.ipynb`）

`data_utils.ipynb` 展示了如何把超大 CSV 流式转成 `.npy`（memmap），并用在线算法统计 mean/std，再生成归一化特征。

当前 notebook 里示例是对 `test.csv` 做的（`len(values)==33`：id + 32 bands）。
你需要**对 train.csv 做同样的事**（train 多一列 label：一般 `len(values)==34`）。

建议的中间产物：

* `resources/features.npy`（float32, shape=[N,32]）
* `resources/labels.npy`（int64, shape=[N]）
* `resources/features_norm.npy`（归一化后的训练特征）
* `resources/test_features_norm.npy`（用训练集 mean/std 归一化后的测试特征）

> ✅ 强烈建议：**用训练集 mean/std 去归一化 test**，不要从 test 重新估计 mean/std（会引入数据泄漏/分布漂移风险）。

---

### Step 1：训练自编码器 AE（`trainer.ipynb`）

`trainer.ipynb` 会：

* 从 `resources/features_norm.npy` 读取训练数据（dataloader 会同时读 labels，但 AE loss 只用 x）
* 训练 `ConvTransformerAE`
* 保存到 `./resources/models/model_{epoch}.pth`

关键配置示例（见 notebook）：

* batch_size=4096
* AdamW(lr=1e-4, weight_decay=1e-4)
* StepLR（注意：代码里 scheduler 是**每个 step**都 `step()`）

运行方式：直接打开 notebook 逐格执行即可。

---

### Step 2：训练分类器（`classifier_train.ipynb`）

`classifier_train.ipynb` 会：

* 构建 `AEClassifier(ae, classifier)`
* 可加载已有权重（示例里 `classifier_model_110000.pth`）
* 冻结/半冻结参数（示例：冻结大部分，只训练分类头 + `ae.to_latent`）
* 使用 **加权交叉熵**缓解类别不均衡
* 在训练中按步保存模型：

  * `./resources/models/classifier_model_{global_step}.pth`

你可以在这里做你们作业要求的“对比实验/消融”：

* 不同冻结策略（全冻AE / 半冻AE / 全量微调）
* 不同重建loss权重（0、0.1、0.2、0.5…）
* 不加权CE vs 加权CE
* 去掉Conv stem / 改 latent_dim / 改 Transformer 深度

---

### Step 3：推理并导出提交文件（`eval.ipynb`）

`eval.ipynb` 会：

* 从 `resources/test_features_norm.npy` 读测试特征
* 加载训练好的 `classifier_model_xxx.pth`
* 输出 `predictions.csv`，列为：

  * `id`
  * `pred_label`

> ⚠️ 本仓库的 `EvalDataset` 默认把 `idx` 当作 `id`。
> 如果你的 `test.csv` 的 `id` 不是从 0 开始连续编号，请在数据预处理阶段把 `id` 单独保存成 `ids.npy`，并在 `EvalDataset` 中返回真实 id。

---

## TensorBoard 日志

* AE 训练日志：`./logs/<exp_name>/...`
* 分类训练日志：`./logs/classifier/<exp_name>/classifier/...`

启动：

```bash
tensorboard --logdir logs
```

---
