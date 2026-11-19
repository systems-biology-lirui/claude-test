# 记忆赛道 v5（Python）

```bash
pip install -r python/memory_v5/requirements.txt
python python/memory_v5/run_v5.py --pack player_data.zip --out output/memory_v5
```

4 核约 4 分钟。输出 `output/memory_v5/谁说bio不如com.zip` 和 `validation.txt`。

## 相对 v4 的改动

只改了 test odor 的解码特征，cue 部分和 behavior 规则与 v4 完全相同（见 `python/memory_v4/README.md`）。

- delay 结束前约 **-780 ms** 有一个 10–30 ms 的尖锐群体瞬变（疑似切换到 test odor 气路时的阀门事件），
  两种 test odor 的瞬变幅度不同。SCm、HPF、DORpm 等脑区反应最强。
- v4 用的是 50 ms 窗口，这个瞬变被平均掉了。v5 读取 delay 末尾 1.2 s 的 **10 ms 分箱**（`data.py` 中的 `d['fine']`），
  新增 3 个 view：`w790_760`、`3win`（-790/-760/-730/-680 ms 三段）和 `fine10`（-800..-700 ms 共 10 个 bin），与 v4 的 test view 合并，k = 5/10/20。
- 门控仍是 inner CV 中 test odor 解码 binomial p < 0.01。更严格的门控（inner 增益 ≥ 2 trial、FA shrinkage、margin）都没有稳定变好。

## 本地外层 CV

| | cue | behavior v4 → v5 | prior |
|---|---|---|---|
| ch1 m091 0608 | 0.987 | 0.703 → 0.700 | 0.665 |
| ch2 m091 0602 | 0.821 | 0.630 → 0.630 | 0.630 |
| ch3 m090 0527 | 0.859 | 0.656 → 0.630 | 0.656 |
| ch3 m095 0709 | 0.974 | 0.729 → **0.813** | 0.625 |

加权 score 0.797 → **0.801**。m095 的 test odor inner 解码从 0.80 提到 0.91。
m090 的下降约等于 32 个 trial 中 0.8 个，属于噪声范围；在模拟中（m090 另一天的 session，只留 32 个标签），新特征让 behavior 从 0.587 提到 0.698。
