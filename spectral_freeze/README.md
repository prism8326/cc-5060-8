# 实时频谱冻结效果器（Spectral Freeze）

基于 STFT + 相位声码器的实时冻结效果。捕获短时间内的**幅度分布与逐频点瞬时频率**，
在停止读取新输入后，按估计频率持续**推进相位**合成稳定、可控的持续音色——
既不是把一小段波形无限循环，也不是用混响尾巴冒充冻结。

## 目录结构

```
spectral_freeze/
  stft.py        流式 STFT / 加权重叠相加（WOLA），捕获窗与输出重叠的实际实现
  dsp.py         包裹相位、瞬时频率、圆形均值、声道相干性、等功率交叉淡化
  engine.py      冻结状态机 LIVE→ENTERING→FROZEN→EXITING、多窗估计、相位推进
  realtime.py    PortAudio 实时回调，与离线共用同一个 process_block
  offline.py     文件离线处理（与实时逐块一致）
  signals.py     自制乐音（含非整周期分音、相位跳变、近静音、直流）与环境音
  gui.py         PySide6 频谱编辑窗口
scripts/
  freeze_file.py       命令行离线处理，可输出 JSON 追踪
  live_freeze.py       实时声卡处理
  make_examples.py     用当前引擎重新生成可听样例与追踪
  benchmark_freeze.py  持续合成成本分解 + 快/参考路径相位与漏更新校验
  verify_consistency.py 实时/离线一致性 + 漂移/波动/瞬态测量
tests/           21 项数值与行为测试
examples/        可听 WAV 样例与验证结果
```

## 安装与启动（界面怎么启动）

普通机器（已有 Python 3.10+）：

```bash
pip install -r requirements.txt
python -m spectral_freeze.gui          # 启动频谱冻结编辑窗口
```

最小化 Debian 容器（无 root、缺 PortAudio/Qt 系统库）时，本仓库提供用户态安装脚本，
它把 Python 包装进 `~/.local`，并把所需 `.so` 下载解压到本地前缀，然后生成 `env.sh`：

```bash
bash setup_env.sh
source env.sh
python -m spectral_freeze.gui
```

无显示环境下可先验证界面可实例化：

```bash
QT_QPA_PLATFORM=offscreen python -c "
from PySide6 import QtWidgets
from spectral_freeze.gui import FreezeWindow
app=QtWidgets.QApplication([]); FreezeWindow().show(); print('GUI OK')"
```

## 界面操作（频谱编辑窗口）

1. 顶部 **Source** 选择自制乐音 / 合成环境录音，或 `Load audio file` 载入 WAV/FLAC。
2. **在波形上点击**：第一次点击放置冻结点，第二次点击放置释放点（可只放冻结点）。
3. 按 **Play** 运行。播放头走到冻结点时自动捕获并进入冻结，走到释放点时平滑退出。
   Play 同时把引擎**实际计算出的输出**送往声卡试听；没有声卡时窗口仍可运行，
   状态栏会说明“静音运行”，左右结果区与状态面板全部来自实时引擎，不是占位数据。
4. 也可随时按 **Freeze now / Release freeze / Reset engine** 手动控制——按钮直接调用
   `FreezeEngine.freeze/unfreeze`，冻结频谱与连续音色都是真实计算结果。
5. 实时参数：进入/退出过渡时长、相干性阈值、音高微调（Hz）、输出增益、
   捕获模式（auto / multi / single）。
6. 结果区显示**冻结幅度谱 + 声道相干性叠加**，可切换到**估计频率轨迹**视图；
   右侧状态面板逐跳显示状态、捕获模式、相位跳变频点数、相干频点数、主要频率、
   live/frozen/output 三路 RMS 与直流幅度。
7. 勾选 `Zero input after freeze point` 后，输入在进入冻结后被清零，
   但扬声器仍持续发声——输出能量完全来自捕获的频谱状态，能量来源可直接追查。
8. 任何处理异常都会在窗口底部状态栏与弹窗中提示，而不是静默失败。

## 关键算法与需求对应

- **捕获窗与重叠规则实际实现**：`stft.py` 用周期性 Hann 窗、hop=256、FFT=2048，
  分析与综合同路径，平方窗能量归一化，直播路径可数值重建（测试误差 < 1e-6）。
- **相位跳变时比较最后瞬间与多窗估计**：`engine.py` 对最近 `capture_windows` 帧做
  解卷绕相位的最小二乘线性外推；稳态峰残差 <0.05 rad、滑音 <0.3 rad、
  真实跳变 >0.5 rad，据此在 auto 模式下选择单窗或多窗捕获。
- **按估计频率推进相位**：冻结后每跳 `phase += IF * hop`（`_advance_frozen`），
  绝不重复同一复数帧，因此没有 hop 速率的周期性脉冲（有专门测试守护）。
- **依声道相干性选续播方式**：相干频点折叠为单声道共同频谱、声像稳定；
  非相干频点保留左右独立相位继续演化。
- **平滑过渡**：进入/退出用等功率交叉淡化，时长可调；测试约束切换处样本阶跃
  不超过直播基线的 10 倍。
- **直流 / 近静音 / 多非整周期分音保护**：直流频点相位增量强制为 0、幅度钳位；
  近静音捕获直接抑制（不产生爆响）；弱能频点 IF 清零防止长时间数值发散；
  分析窗预热满之前不允许捕获，避免把启动零样本冻进去。
- **停止输入后行为可追查**：每跳生成 `FrameReport`（状态、三路 RMS、频率、捕获信息），
  CLI 可导出完整 JSON；`verify_consistency.py` 证明输入 RMS=0 时输出仍稳定可闻。

## 持续合成提速（冻结态计算顺序）

冻结是最贵的稳态——输入可以永久为 0，冻结音却要一直发声。提速只改冻结态，逐步削减
**重复分析**与**无意义谱运算**，同时保留真实的相位推进与声道关系：

1. **跳过冻结态的分析 FFT**：完全冻结时不再每跳做一次 `rfft`，只把分析缓冲按 hop
   前移（`STFT.push`）。退出时状态先变为 `EXITING`，下一跳照常 `analysis`，因此退出
   路径与原始实现一致；`unfreeze` 会清空历史，重新冻结需重新预热，不会用陈旧频谱捕获。
2. **只在活跃分音上更新相位**：估计频率恰为 0 的频点相位永远不变，其复数谱值
   `mag·exp(jφ)` 在捕获时算一次即可复用；每跳只对约 5% 的活跃分音做
   `wrap_phase` 与 `exp`。这是真实的逐分音相位推进，不是循环播放固定波形。
3. **相干掩码只在阈值变化时重建**：拖动相干阈值滑杆不会每跳重算比较。
4. **报告 RMS 不再额外做 IFFT**：用 Parseval 从频谱直接得 RMS；冻结 RMS 与相位无关，
   捕获时缓存一次。
5. **全局音高微调（drift）**：一旦有全局 nudged，所有频点都会移动，此时自动退回逐频点
   推进，静态复用关闭，行为与参考实现一致（有测试守护）。

`engine.py` 保留 `reuse_cache=False` 的原样**参考路径**（每跳对每个频点重算），用于
交叉校验。`scripts/benchmark_freeze.py` 在长持续音（输入清零）与参数拖动两种负载下
分别测量 CPU，并逐步比较快/参考两条路径的**逐频点相位差**与**漏更新数**（应为 0）：

```bash
python -m scripts.benchmark_freeze     # 生成 examples/freeze_benchmark.json
git worktree add /tmp/baseline HEAD    # 可选：同机对照原始实现
python -m scripts.benchmark_freeze \
  --baseline-src /tmp/baseline/spectral_freeze/spectral_freeze
python -m scripts.make_examples        # 重新生成可听样例与 phasejump_trace.json
```

实测（48 kHz、hop=256，同一台机器）：冻结稳态从未优化的约 154 µs/跳降到约 67 µs/跳
（约 2.31×），参数拖动负载约 157 → 70 µs/跳（约 2.26×）；快/参考路径逐频点最大相位差
< 1e-16 rad、漏更新 0、输出逐样本差 < 3e-16，冻结频率漂移 0 Hz（`reuse_cache=False`
参考路径与 `reuse_cache=True` 快路径的差异仅为浮点舍入）。`examples/freeze_benchmark.json`
记录本次实测的成本分解（合成、相位推进、报告、分析前移）与原始实现对照。
真实声卡不可用时命令行 `live_freeze.py` 仅作辅助，界面仍是主要入口。

## 离线处理与验证

```bash
# 冻结并在输入停止后继续发声，输出 JSON 追踪
python -m scripts.freeze_file --input examples/musical_input.wav \
  --output out.wav --freeze-at 0.9 --release-at 2.6 --stop-input-at 1.3 \
  --report trace.json

# 实时/离线一致性与质量测量（无声卡也可运行，回调被逐块模拟驱动）
python -m scripts.verify_consistency

# 单元测试
python -m pytest tests/ -q
```

`examples/verification.json` 记录的实测结论：离线与实时回调输出最大差为 0；
冻结频率峰峰漂移为 0 Hz；冻结段幅度变异系数 < 0.18；进入/退出瞬态低于约束；
输入归零后输出 RMS 仍在 0.038–0.25，可闻且仅来自冻结态。

## 可听样例（examples/）

| 文件 | 说明 |
| --- | --- |
| `musical_input.wav` / `ambient_input.wav` | 自制乐音与短环境录音 |
| `musical_freeze.wav` | 稳态单窗捕获、平滑进入退出 |
| `musical_freeze_phasejump.wav` | 冻结点落在 180° 相位跳变上，auto 选择多窗估计 |
| `musical_freeze_nearsilence.wav` | 在近静音间隙冻结，捕获被抑制、无爆响 |
| `musical_freeze_indefinite.wav` | 1.3 s 后输入永久清零，冻结音无限持续 |
| `ambient_freeze.wav` | 环境音冻结 |
| `phasejump_trace.json` / `verification.json` | 逐跳追踪与量化验证 |
| `freeze_benchmark.json` | 持续合成提速：成本分解、快/参考相位差与漏更新、CPU |

## 实时声卡

> 命令行入口仅辅助频谱冻结（批量处理、验证、无界面环境）；频谱冻结的图形入口是
> `python -m spectral_freeze.gui`，界面里在波形上点击并拖动过渡滑杆即可听到冻结音。

```bash
python -m scripts.live_freeze --list-devices
python -m scripts.live_freeze --input 1 --output 2 --entry-fade 0.08
# 运行中回车切换 冻结/释放，Ctrl-C 退出；欠载次数在退出时报告
```

实时回调与离线、GUI 调用的是同一个 `FreezeEngine.process_block`，因此三条路径
输出逐样本一致（已由测试固定）。
