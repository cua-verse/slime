# 环境 Abort 设计方案

---

## 重要约束

**只允许修改 `examples/geo3k_vlm_multi_turn_unfold` 里面的文件。**

若需修改本目录以外的文件，必须先复制到本目录下再修改。

---

## 文件结构

abort env 功能通过 **独立文件** 实现，不影响原有逻辑：

```
examples/geo3k_vlm_multi_turn_unfold/
├── rollout.py                                          # 原有，不改动
├── run_geo3k_vlm_multi_turn_unfold.sh                  # 原有，不改动
├── run_geo3k_vlm_multi_turn_unfold_async.sh            # 原有，不改动
├── rollout_abort_env.py                                # 新建：abort env 的 generate + rollout 入口
├── run_geo3k_vlm_multi_turn_unfold_abort_env.sh        # 新建：使用 abort env 的训练脚本
└── ENV_ABORT_DESIGN.md                                 # 本文档
```

- 使用原版：`run_geo3k_vlm_multi_turn_unfold.sh` → `rollout.py` → 默认 `generate_rollout`
- 使用 abort env 版：`run_geo3k_vlm_multi_turn_unfold_abort_env.sh` → `rollout_abort_env.py` → 自定义 `generate_rollout`

---

## 背景

- `abort()`（`slime/rollout/sglang_rollout.py:307`）只做两件事：(1) 设 `state.aborted = True`，(2) 向 SGLang worker 发 `POST /abort_request`，然后等待 pending 任务结束。
- 当前 `generate()`（`rollout.py:229`）循环中 **没有检查 `state.aborted`**，即使 abort 已触发也会继续做无用功。
- 对于轻量 env（如 Geo3kEnv），`finally: env.close()` 已足够。但对重量级 env（VM、容器、长驻进程），需要在 abort 时 **主动** 清理资源。

---

## 现有 abort 流程

```
generate_rollout_async()                    # sglang_rollout.py:351
  └─ 收集够样本后 ─► abort(args, rollout_id)  # sglang_rollout.py:423
       ├─ state.aborted = True               # :312
       ├─ POST /abort_request → SGLang       # :322
       └─ while pendings: await              # :330  等所有 generate task 完成
            └─ generate() finally: env.close()
```

关键观察：
1. **推理中 abort**：`generate()` 阻塞在 `_run_text_inference` → SGLang abort 使其返回 `finish_type="abort"` → break → finally → `env.close()`。**env 会被清理。**
2. **env.step() 中 abort**：SGLang abort **无法中断** `env.step()`。必须等 step 返回后在下一轮检查才能退出。
3. **框架入口**：`--rollout-function-path`（默认 `slime.rollout.sglang_rollout.generate_rollout`）可指定自定义 rollout 入口函数，从而控制整个 rollout 循环和 abort 流程。
4. **GenerateState 是 Singleton**（`SingletonMeta`），不能复制类定义，只能 import 使用原始类。

---

## `rollout_abort_env.py` 设计

该文件是自包含的一体化模块，包含三部分功能：

### 1. 带协作检查的 `generate()`

从 `rollout.py` 的 `generate()` 基础上增加 `state.aborted` 检查和 env 注册/反注册：

```python
# ---- 模块级 env 注册表 ----
_active_envs: set = set()

def register_env(env):
    _active_envs.add(env)

def unregister_env(env):
    _active_envs.discard(env)


async def generate(args, sample, sampling_params) -> list[Sample]:
    ...
    state = GenerateState(args)
    ...
    env = _build_env(...)
    register_env(env)
    try:
        env.reset()
        for turn_idx in range(max_turns):
            # ---- 协作式检查 ----
            if state.aborted:
                break

            response_text, ... = await _run_text_inference(...)
            ...
            observation, done, _info = env.step(response_text)
            if done:
                ...
                break

            # ---- step 后再检查一次 ----
            if state.aborted:
                break
            ...

        if state.aborted and turn_samples:
            turn_samples[-1].status = Sample.Status.ABORTED

        return turn_samples
    finally:
        unregister_env(env)
        try:
            env.close()
        except Exception:
            pass
```

### 2. 带 env 清理的 `abort()`

从 `sglang_rollout.abort` 复制（约 40 行），在 `state.aborted = True` 之后、SGLang abort 之前，插入 env 清理：

```python
async def abort(args, rollout_id):
    state = GenerateState(args)
    state.aborted = True  # 注意：去掉原版的 assert not state.aborted

    # ---- 新增：主动清理所有已注册的 env ----
    for env in list(_active_envs):
        try:
            if hasattr(env, 'abort'):
                env.abort()
            else:
                env.close()
        except Exception:
            logger.warning("Failed to abort env %s", env, exc_info=True)

    # ---- 以下与 sglang_rollout.abort 相同 ----
    # POST /abort_request 到所有 SGLang worker
    # await pendings
    ...
```

### 3. 自定义 `generate_rollout` 入口

从 `sglang_rollout.generate_rollout_async` + `generate_rollout` 复制（约 60 行），仅将 `abort` 调用替换为本模块的版本：

```python
async def generate_rollout_async(args, rollout_id, data_source):
    """与 sglang_rollout.generate_rollout_async 相同，仅 abort 指向本地版本。"""
    ...
    aborted_samples = await abort(args, rollout_id)  # 本地的 abort
    ...

def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """入口函数，签名与 sglang_rollout.generate_rollout 一致。"""
    if evaluation:
        # eval 不需要 env abort，直接用原版
        from slime.rollout.sglang_rollout import eval_rollout
        output, _ = run(eval_rollout(args, rollout_id))
        return output
    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
```

**注意**：
- **不复制 `GenerateState`**——它是 Singleton，直接 import 原始的。
- **不复制 `generate_and_rm`、`generate_and_rm_group`**——它们由 `GenerateState.submit_generate_tasks` 调用，无需改动。
- 只复制 `abort` + `generate_rollout_async` + `generate_rollout`（共约 100 行），加上带协作检查的 `generate`。
- `convert_samples_to_train_data` 从 `rollout.py` 直接复用（import 或复制均可）。

---

## `run_geo3k_vlm_multi_turn_unfold_abort_env.sh` 设计

从 `run_geo3k_vlm_multi_turn_unfold_async.sh` 复制，仅修改两处参数：

```bash
# 1. generate 函数指向 abort_env 版本
--custom-generate-function-path examples.geo3k_vlm_multi_turn_unfold.rollout_abort_env.generate

# 2. rollout 入口指向 abort_env 版本（控制 abort 流程）
--rollout-function-path examples.geo3k_vlm_multi_turn_unfold.rollout_abort_env.generate_rollout
```

其余参数（模型、数据集、超参）与 async 版完全一致。

---

## env 接口约定

当前 `Geo3kEnv` 无需改动（`close()` 是空操作，`step()` 瞬时完成）。

未来重量级 env 需遵循：
- `abort()`：可选方法，强制终止（杀进程、关容器等）。若不提供，`rollout_abort_env.py` 会 fallback 到 `close()`。
- `close()`：幂等，abort 后再次调用不应报错。
- `step()`：若可能长时间阻塞，应设计为 async 或在 executor 中运行，否则 abort 中的 `env.abort()` 无法及时执行（asyncio 单线程限制）。

---

## 实施计划

| 步骤 | 改动 | 文件 |
|------|------|------|
| 1 | 创建 `rollout_abort_env.py`：带协作检查的 generate + env 注册表 + 带 env 清理的 abort + 自定义 generate_rollout 入口 | `rollout_abort_env.py`（新建） |
| 2 | 创建训练脚本，从 async.sh 复制并修改两处路径参数 | `run_geo3k_vlm_multi_turn_unfold_abort_env.sh`（新建） |

**仅新增 2 个文件，原有文件零改动。**

---

## 边界情况

| 场景 | 行为 |
|------|------|
| generate 在推理中 | SGLang abort → 返回 abort → break → finally close |
| generate 在 env.step() 中（瞬时） | step 返回 → 协作检查 break + abort 中主动 env.abort() |
| generate 在 env.step() 中（长阻塞） | abort 中主动 env.abort() 中断 step |
| generate 在两次推理之间 | 协作检查 → break → finally close |
| 重复 close/abort | env.abort() + env.close() 均应幂等 |
| 并发安全 | asyncio 单线程，遍历用 `list()` 快照 |
