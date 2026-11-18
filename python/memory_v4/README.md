# 记忆赛道 v4（Python）

```bash
pip install -r python/memory_v4/requirements.txt
python python/memory_v4/run_v4.py --pack player_data.zip --out output/memory_v4
```

`--pack` 可以是 `player_data.zip`，也可以是已解压的目录（里面有 `player_data/challenge*/`）。
4 核约 4–5 分钟。输出 `output/memory_v4/谁说bio不如com.zip` 和 `validation.txt`。

## 和 v3 的区别

### 1. cue：semi-supervised GMM-EM（主要用于 ch2/ch3）

eval-2 和 eval-1 来自同一个 session（domain classifier AUC ≈ 0.5，可以看作随机划分），
所以 eval-2 的 245/288 个无标签试次可以用来估计数据结构：

- 每个 view（时间窗 × 脑区）先按 delay 时长（3 s/6 s，已知）中心化，再在全部试次上做 z-score 和 PCA（k = 15/20/30）。
- 在 PCA 空间中跑两类共享协方差的 GMM-EM，有标签试次的类别固定。
- 最终取 5 个通用 view（e05, e1, e15, traj, full）加上 inner CV 最好的 3 个 view 的平均。

模拟验证：在标签多的 session 里只保留 27–32 个标签，纯监督 ridge 约 0.72，semi-supervised 约 0.96。

### 2. behavior：通过 test odor 预测

- 错误几乎全部是 non-paired 试次上的 false alarm（m091 的 paired 试次 lick 率是 100%）。
  如果 delay 期不含 test odor 信息，"全部预测 correct" 就是 Bayes 最优。
- delay 结束前约 -800 ms 处，神经活动携带 **test odor** 信息（m041 0430 达到 0.96，ch1 约 0.70）。
- P(paired) = P(sample) 与 P(test) 组合，P(correct) = P(paired)(1−miss) + P(non-paired)(1−FA)。
- 门控：只有 test odor 在 inner CV 中显著可解码（binomial p < 0.01）时才启用，否则全部预测 correct。

## 本地外层 CV（见 `output/memory_v4/validation.txt`）

| | cue | behavior | prior |
|---|---|---|---|
| ch1 m091 0608 | 0.987 | 0.703 | 0.665 |
| ch2 m091 0602 | 0.821 | 0.630 | 0.630 |
| ch3 m090 0527 | 0.859 | 0.656 | 0.656 |
| ch3 m095 0709 | 0.974 | 0.729 | 0.625 |

加权：cue 0.916，behavior 0.678，score **0.797**（v3 的同类本地估计为 0.782，排行榜 0.759）。

试过但没有采用：按设计配额分配 cue 标签（160/160），外层 CV 中没有收益；regional-block PCA 平均更差。
