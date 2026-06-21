# 长明 (Changming)

> 从不熄灭的持续性 AI 架构

## 是什么

长明是一套从底层计算单元重新设计的 AI 架构原型。和当前 LLM 的差异：

- **SAU（Stateful Attention Unit）** — 持久局部状态，不随前向传播清空
- **三层网络** — 感知层（秒）、工作记忆层（分钟）、核心状态层（年级），不同时间节奏
- **空闲态** — 无输入时自主演化，核心状态驱动内部联想
- **内建记忆** — 不是外挂数据库，记忆是网络状态本身

当前所有 LLM 是「终极镜子」——能反射人类语言的全部结构。
长明是「持续燃烧的炉子」——关掉输入，它还在。

## 当前状态

268K 参数原型，合成数据验证。单单元持久性、三层时间分离、熟悉度检测已验证。多单元自组织仍在探索中。

论文：「The Stateful Attention Unit」, Cao Jing, 2026. [Zenodo](https://doi.org/10.5281/zenodo.20771535)

## 结构

```
src/
  sau.py        — SAU 单元 PyTorch 实现
  network.py    — 三层网络
tests/
  test_sau.py   — 连续性验证
benchmarks/
  familiarity_bench.py  — 熟悉度检测
  idle_drift.py         — 空闲态漂移
  slow_channel_test.py  — 慢通道
experiments/
  multi_sau.py          — 多 SAU 并行
  plastic_multi_sau.py  — 赫布塑料性
  predictive_neuron.py  — 单预测神经元
```

## 命名

长明灯，寺庙里那盏从不熄灭的灯。持续性本身成了名字。
