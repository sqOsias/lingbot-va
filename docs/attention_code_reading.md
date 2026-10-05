# 注意力机制代码阅读指南

本文按当前仓库源码梳理 LingBot-VA2 的 Transformer 注意力实现。主入口是 [`wan_va/modules/model.py`](../wan_va/modules/model.py)；训练调用见 [`wan_va/train.py`](../wan_va/train.py)，在线推理调用见 [`wan_va/wan_va_server.py`](../wan_va/wan_va_server.py)。阅读时建议先看 `WanTransformer3DModel.forward_train/forward`，再看 `WanTransformerBlock.forward`、`WanAttention.forward` 和 `FlexAttnFunc`。

## 1. 一张调用图

```text
训练：Trainer._prepare_input_dict
  → WanTransformer3DModel.forward(train_mode=True)
  → forward_train：拼接四类 token，建立 FlexAttention 掩码
  → 每个 WanTransformerBlock：self-attention → text cross-attention → FFN
  → 只取噪声态 latent/action token 输出预测

推理：WanVAServer._infer / _compute_kv_cache
  → WanTransformer3DModel.forward(train_mode=False)
  → 每个 WanTransformerBlock：self-attention（可用 KV 缓存）
                            → text cross-attention → FFN
  → 视频与 action 分两次调用模型
```

模型默认有 30 个 block、24 个头、每头 128 维，所以注意力内部宽度为 `24 × 128 = 3072`；这些是 [`WanTransformer3DModel.__init__`](../wan_va/modules/model.py) 的默认参数，实际运行还受加载的模型配置影响。每个 block 有独立的 `attn1`（自注意力）和 `attn2`（文本交叉注意力）。

## 2. 单层注意力如何计算

核心在 [`WanAttention`](../wan_va/modules/model.py)：

1. `to_q/to_k/to_v` 分别线性投影，`norm_q/norm_k` 对展平的多头维度做 RMSNorm；V 不做这个归一化。
2. 把 `[B, L, heads × head_dim]` 拆成 `[B, L, heads, head_dim]`。自注意力的 Q/K/V 来自同一视频/action token 序列；交叉注意力的 Q 来自当前序列，K/V 来自投影后的文本 token。
3. 若传入 `rotary_emb`，只旋转 Q 和 K。随后调用由 `attn_mode` 选定的算子，结果合并头维，经线性层和 dropout 输出。

算子输入均按 `[B, L, H, D]` 准备，三种模式的区别如下：

| `attn_mode` | 算子 | 这里的掩码/形状处理 | 主要使用处 |
| --- | --- | --- | --- |
| `torch` | `torch.nn.functional.scaled_dot_product_attention` | `custom_sdpa` 转成 `[B,H,L,D]` 后调用；没有传入显式掩码 | 服务器推理显式指定 |
| `flashattn` | `flash_attn_func` | 直接接收 `[B,L,H,D]`；这里没有传入显式掩码 | 可选推理模式 |
| `flex` | 编译后的 `flex_attention` | `FlexAttnFunc` 使用预建的自注意力/交叉注意力 `BlockMask`；把展平的训练 token 重新摆成 `[1,H,L,D]` | 训练显式指定 |

`FlexAttnFunc.forward` 读取 `query[0]`，因此对应的是训练路径把原始 batch 展平成一个长序列、外层 batch 维固定为 1 的布局。它把非半精度 Q/K/V 转为 `bfloat16`（默认），并设置 FlexAttention 的 block kernel 选项。`attention_mask` 和 `cross_attention_mask` 是类变量，在每次 `forward_train` 中重建；这也意味着同一进程内并发运行不同长度/规则的训练前向时，不能把它们当作彼此独立的实例状态。

## 3. 位置与条件怎样进入注意力

[`WanRotaryPosEmbed`](../wan_va/modules/model.py) 将每头维度分给时间 `f`、高度 `h`、宽度 `w` 三轴。默认每头 128 维时分别为 44、42、42 维。它由 `grid_id` 的前三行算出复数相位，再在 `WanAttention.forward` 对 Q/K 成对维度作复数旋转。`grid_id` 的第 4 行是 latent/action 类型标记，但 RoPE 的 `forward` 只读取前三行。

[`get_mesh_id`](../wan_va/utils/utils.py) 为视频建立 `(frame, row, col)` 坐标；action 的帧坐标还加上同帧内 action 位置的小数偏移，高/宽坐标置为 `-1`。训练通过 `_add_noise` 建网格；推理在 `_prepare_latent_input` 中传入 `frame_st_id`，使连续块的帧坐标递增。视频先按 `patch_size` 展平为 token，action 则以其 `F × H × W` 位置为 token。

扩散时间步经 `WanTimeTextImageEmbedding` 映射为每个 token 的调制量。每个 block 用其中六组 shift/scale/gate 调制自注意力和 FFN 的归一化与残差。文本先经 `text_embedder` 投影，再作为每层交叉注意力的 K/V；**文本交叉注意力不使用此处的三轴 RoPE**。

## 4. 训练：FlexAttention 的可见性规则

[`forward_train`](../wan_va/modules/model.py) 将 token 按以下顺序拼接：

```text
[噪声视频 latent, 条件视频 latent, 噪声 action, 条件 action, padding]
```

原始 batch 的 token 在序列维展平；序列长度补到 128 的倍数。`FlexAttnFunc.init_mask` 为每个 token 生成样本编号 `seq_id`、块编号 `frame_id` 和状态编号 `noise_id`。`noise_id=0` 表示噪声态，`1` 表示条件态（代码中的 `clean`）；padding 的 `seq_id=-1`，不会被注意力选中。视频块编号为 `2 × floor(frame/chunk_size)`，action 块编号为 `2 × floor(frame/chunk_size)+1`，于是一个时间块内先视频、后 action。`chunk_size` 和 `window_size` 由训练输入提供；普通训练入口在 [`_prepare_input_dict`](../wan_va/train.py) 中分别随机取 1–4、4–64。

自注意力允许的 Q→K 关系是：

| Q 的状态 | K 的状态 | 块编号条件 |
| --- | --- | --- |
| 条件态 | 条件态 | `key.frame_id ≤ query.frame_id` |
| 噪声态 | 条件态 | `key.frame_id < query.frame_id` |
| 噪声态 | 噪声态 | `key.frame_id == query.frame_id` |

三种情况还必须同时满足**同一个样本**、非 padding，并且 `abs(query.frame_id - key.frame_id) ≤ window_size`。所以条件态可以看同块及窗口内的历史条件态；噪声态可看同块噪声态以及窗口内更早的条件态，不能看同块条件态或其他块的噪声态。由于视频块编号为偶数、action 为其后的奇数，这套规则也规定了同时间块内的模态顺序。这里的“窗口”单位是上述交错块编号，并非原始帧数。

文本交叉注意力的 mask 仅要求 Q 与文本 K 属于同一个原始样本、Q 非 padding。文本侧按每个样本 **512 个 token** 建立编号，代码没有在这层使用 tokenizer 的逐 token `attention_mask`。因此该实现依赖训练输入的文本长度与此处固定长度一致。输出阶段仅拆取两段噪声态 token，经视频/action 各自输出头得到训练预测。

## 5. 推理：自注意力 KV 缓存

服务器在 [`_reset`](../wan_va/wan_va_server.py) 中按 `attn_window // 2` 个视频块和同数 action 块的 token 容量，为**每层自注意力**创建 K/V 池；交叉注意力不创建缓存。`WanAttention` 的缓存含 `k`、`v`、递增的块 `id`、有效位 `mask` 和预测位 `is_pred`。一个前向调用会先将当前 Q 对应的 K/V 放入池，再让 Q 对池中所有有效 K/V 做注意力；当前调用内部的 token 也彼此可见。

`update_cache` 控制调用结束后是否留下刚写入的 K/V：

| 值 | 结果 | 调用位置 |
| --- | --- | --- |
| `0` | 当前 K/V 临时入池供本次计算，结束后撤销有效位 | 非最后一次去噪迭代 |
| `1` | 保留当前 K/V，标记为预测 token | 视频或 action 去噪的最后一次迭代 |
| `2` | 保留当前 K/V，标记为已观测 token | `_compute_kv_cache` 编码新观测 |

在新观测到来时，`_compute_kv_cache` 先 `clear_pred_cache`，移除先前预测 token，再依次写入观测视频和 action。空间不足时 `allocate_slots` 按缓存 `id` 淘汰最早的有效 token；这个 `id` 是**每次写入递增的调用编号**，不是 RoPE 的帧编号。推理算子自身没有传因果或滑窗 mask，历史范围由缓存容量和上述更新顺序决定。服务器先运行视频去噪、再运行 action 去噪，因此后者可读到已保留的视频 K/V。

## 6. 配置与阅读时的边界

- [`train.py`](../wan_va/train.py) 加载模型时显式传 `attn_mode="flex"`；[`wan_va_server.py`](../wan_va/wan_va_server.py) 显式传 `attn_mode="torch"`。两者经 [`load_transformer`](../wan_va/modules/utils.py) 把参数交给 `from_pretrained`。检查实际模式应以调用处和最终加载配置为准。
- `forward_train` 无条件建立 Flex mask，设计上与 Flex 算子配合。普通推理 `forward` 不调用 `init_mask`，因此不能直接用 Flex 模式走该路径；这与 [`README.md`](../README.md) 的配置提示一致。
- 训练 mask 使用 `seq_id` 隔离合并后的样本；推理采用原生 batch 维（开启 CFG 时为 2），两条路径的 batch 布局不同。
- `model.py` 在模块导入时尝试导入 FlashAttention（先 `flash_attn_interface`，失败则 `flash_attn`）；即使选择 `torch` 模式，也要留意这个导入依赖。
- 默认 `patch_size=[1,2,2]`。`init_mask` 的视频 `seq_id` 数量按时间 patch 后长度计算，而 `latent_frame_id` 用原始 `L_F` 构造；若改动时间 patch 大小，应先核对二者长度及 token 对齐。

快速定位：`FlexAttnFunc._get_mask_mod` 看训练可见性；`WanAttention.forward` 看 QKV、RoPE 和缓存；`WanTransformerBlock.forward` 看层内残差与文本交叉注意力；`WanTransformer3DModel.forward_train/forward` 看训练、推理输入布局。
