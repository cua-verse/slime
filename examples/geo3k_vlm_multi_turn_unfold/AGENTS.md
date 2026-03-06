# geo3k_vlm_multi_turn_unfold

多轮 VLM rollout 的"展开"版本：第 n 轮生成时，context 中前 n-1 轮的 `<think>...</think>` 内容被剥离。

## 动机

标准多轮 rollout 在第 n 轮生成时，模型可以看到前序所有轮次的完整 `<think>` 内容。
这使得后续轮次的推理可以直接复用前轮的隐藏状态，而不是真正地 "重新思考"。

目标：强迫模型在每个新轮次基于干净的结论（无前序推理过程）重新推理。

## 设计

### 用户直觉（正确）

不能只改 rollout 生成逻辑：

- **生成时** 剥离 think：容易，改 context 构建即可
- **训练时** 需要一致的 context（训练时 context ≠ 生成时 context → importance sampling 偏差）

正确做法：一条 K 轮轨迹**展开为 K 个独立训练 Sample**（unfold），每个 Sample 的 context 就是生成时实际看到的 stripped 序列。

### 每轮 Sample 结构

```
Turn k 的训练 Sample:
  context_k   = [prompt_ids] + [strip_think(resp_1)] + [obs_1] + ... + [strip_think(resp_{k-1})] + [obs_{k-1}]
  tokens      = context_k + [resp_k]
  loss_mask   = [1 × len(resp_k)]
  response_length = len(resp_k)
  rollout_log_probs = actual_log_probs_k
```

- `context_k` 整体作为训练框架的"prompt"，满足 `total_length = len(context_k) + response_length`
- `prompt_ids`：由 HuggingFace processor 处理（含展开的 image pad tokens）
- `strip_think(resp)` 使用正则 `<think>.*?</think>` 剥离完整 think 块，`<think>.*$` 剥离末尾截断块
- 当前轮的 `<think>` 保留（loss_mask=1，正常训练）
- 返回值：`list[Sample]`，slime 基础设施已原生支持（`sglang_rollout.py:252`）

### SGLang IndexError 修复

**Bug 根因**：HF processor 把 1 个 `<|image_pad|>` 占位符展开为 N 个 token；将这 N 个 token 作为 `input_ids` 发给 SGLang 时，SGLang 解码回文本得到 N 个 `<|image_pad|>` 字符串，误认为有 N 张图 → `image_grid_thw[1]` IndexError。

**修复**：在 payload 中用 `"text"` 替代 `"input_ids"`。文本中只含 1 个 `<|image_pad|>` 占位符，SGLang 内部做正确的 expand。

```python
# 修复前（有 bug）
payload["input_ids"] = prompt_ids  # 含 N 个 image_pad tokens

# 修复后
payload["text"] = context_text     # 含 1 个 <|image_pad|> 占位符
```

对于多轮（k > 1），context_text 由文本拼接构建，同样只含原始占位符数量，不会重复扩展。

### GRPO Advantage 计算

```
一条轨迹 (K 轮) → K 个 per-turn Samples，共享同一个 advantage

advantage 在轨迹级别计算：
  - 每条轨迹的 reward = 最后一轮的 terminal reward
  - 同一 group（n_samples_per_prompt 条轨迹）内做 mean/std 归一化
  - 所有 per-turn Samples 分配同一 advantage
```

通过 `custom_convert_samples_to_train_data_func` 实现，在 `rollout.py` 中定义：
`convert_samples_to_train_data(args, samples) -> train_data_dict`

## 文件结构

```
examples/geo3k_vlm_multi_turn_unfold/
├── AGENTS.md                              # 本文件
├── __init__.py
├── rollout.py                             # generate() + convert_samples_to_train_data()
├── geo3k_vlm_multi_turn_unfold_config.yaml
└── run_geo3k_vlm_multi_turn_unfold.py     # 启动脚本
```

env 和 base_env 直接复用 `examples.geo3k_vlm_multi_turn` 中的实现。

## 与 geo3k_vlm_multi_turn 的差异

| 方面 | geo3k_vlm_multi_turn | geo3k_vlm_multi_turn_unfold |
|------|----------------------|------------------------------|
| SGLang payload | `input_ids` (有 bug) | `text` (fixed) |
| 前序 think 可见性 | 可见 | 不可见（剥离） |
| generate() 返回值 | `Sample` | `list[Sample]` |
| 训练 Sample 数量 | 1 per rollout | K per rollout |
| convert_samples 函数 | 默认 | 自定义（处理 per-turn unfold）|
| GRPO 归一化粒度 | per-sample | per-rollout (terminal reward) |

## 验证方法

1. 每轮 generation 的 context 中不含前序 `<think>` 内容（打印验证）
2. per-turn Sample 的 `loss_mask` 前缀为 0，当前 turn response 为 1
3. `len(loss_mask) == response_length` assertion 通过
4. 同一轨迹的所有 per-turn Samples 分配相同 advantage
5. training batch 正常通过，无 shape 错误
