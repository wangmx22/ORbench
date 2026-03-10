# ORBench

**通用 CPU→CUDA 加速 Benchmark：评测大语言模型的 GPU 编程能力**

ORBench 评测大语言模型能否将通用 CPU 代码自动改写为高效的 CUDA kernel。与现有 benchmark 不同——KernelBench 只关注 ML 算子，ComputeEval 只考察 CUDA 语法正确性——ORBench 覆盖经典算法，包括图遍历、动态规划、排序、组合优化等。

## 快速开始

```bash
# 1. 列出所有可用任务
python run.py list

# 2. 生成测试数据
python tasks/bellman_ford/gen_data.py large tasks/bellman_ford/data/large

# 3. 用 LLM 生成 CUDA 代码（需要 API key）
export LLM_API_KEY="your-api-key"
python run.py generate --task bellman_ford --model claude-sonnet-4-20250514 --level 2 --samples 3

# 4. 评测（需要 NVIDIA GPU）
python run.py eval --run claude-sonnet-4-20250514_l2 --arch sm_89

# 5. 评测 + 保存完整 nsys 分析报告
python run.py eval --run claude-sonnet-4-20250514_l2 --arch sm_89 --save-nsys

# 6. 查看结果
python run.py analyze --run claude-sonnet-4-20250514_l2
```

## 项目结构

```
ORBench/
├── run.py                      # 统一 CLI 入口
├── config.yaml                 # 全局配置（模型、GPU 架构、超时等）
├── requirements.txt
│
├── framework/                  # 评测框架
│   ├── task.py                 # 任务加载、TaskConfig 数据类
│   ├── generate.py             # 调 LLM API 生成 .cu 代码
│   ├── compile.py              # nvcc 编译（支持 CPU 并行）
│   ├── validate.py             # 多规模正确性验证（对比 .bin 文件）
│   ├── benchmark.py            # CUDA Event 计时 + nsys 集成
│   ├── profile.py              # nsys 自动化：profile → 导出全部 CSV → 分析
│   ├── batch_eval.py           # 多 GPU 批量调度（spawn 进程隔离）
│   └── analyze.py              # 结果汇总：fast_p、按类别统计
│
├── tasks/                      # 任务定义（以 bellman_ford 为模板）
│   └── bellman_ford/
│       ├── task.json           # 任务元信息（难度、输入规模、容差、优化点）
│       ├── prompt_l1.md        # Level 1 prompt：CPU 代码 + 算法描述 + GPU 优化提示
│       ├── prompt_l2.md        # Level 2 prompt：CPU 代码 + 算法描述
│       ├── prompt_l3.md        # Level 3 prompt：仅 CPU 代码
│       ├── LLM_input.cu        # LLM 填写模板（与 cpu_reference.cu 结构完全一致）
│       ├── cpu_reference.cu    # CPU 基准实现
│       ├── gen_data.py         # 测试数据生成器（输出 .bin 二进制文件）
│       ├── compare.py          # 正确性验证（二进制浮点对比 + 容差）
│       └── data/
│           └── large/          # 预生成的测试数据
│               ├── input.txt           # V E source seed
│               ├── row_offsets.bin     # CSR 图结构（int32）
│               ├── col_indices.bin
│               ├── weights.bin         # 边权（float32）
│               ├── expected_dist.bin   # CPU 参考答案（float32）
│               └── meta.json           # 文件格式说明
│
├── runs/                       # LLM 生成的代码 + 评测结果
│   └── <run_name>/
│       └── bellman_ford/
│           ├── sample_0.cu             # LLM 生成的代码
│           ├── nsys_sample_0/          # nsys 分析结果（--save-nsys 时生成）
│           │   ├── nsys_summary.txt
│           │   ├── nsys_cuda_gpu_trace.csv
│           │   ├── nsys_cuda_gpu_kern_sum.csv
│           │   ├── nsys_cuda_gpu_mem_time_trace.csv
│           │   ├── nsys_cuda_gpu_mem_time_sum.csv
│           │   ├── nsys_cuda_gpu_mem_size_sum.csv
│           │   ├── nsys_cuda_api_trace.csv
│           │   └── nsys_cuda_api_sum.csv
│           └── ...
│
└── cache/                      # 编译缓存
```

## 评测流程

```
任务定义 → LLM 生成代码 → nvcc 编译 → 正确性验证 → 性能计时 → 结果分析
                                          ↓                ↓
                                    对比 .bin 文件    CUDA Event（必须）
                                    多规模测试        nsys trace（可选）
```

### LLM 的任务

LLM 收到 `LLM_input.cu` 模板，只需填写两个 `>>> LLM CODE START/END <<<` 之间的部分：

1. **CUDA kernel 和 device 函数**
2. **`gpu_bellman_ford()` 函数体**：cudaMalloc → cudaMemcpy → kernel launch → 结果回拷 → cudaFree

模板中的 main 函数负责：读 .bin 输入、warmup 3 次、CUDA Event 计时 10 次、打印 `GPU_TIME_MS`、`--validate` 模式写 output.bin。LLM 不需要碰这些。

### 手动编译运行

```bash
# 编译
nvcc -O2 -arch=sm_89 -o solution LLM_output.cu

# 计时（秒出）
./solution tasks/bellman_ford/data/large

# 验证正确性
./solution tasks/bellman_ford/data/large --validate

# nsys profiling
nsys profile -t cuda -s none -o report ./solution tasks/bellman_ford/data/large
nsys stats --force-export=true --report cuda_gpu_trace --format csv -o report report.nsys-rep
```

## 评测指标

| 指标 | 说明 | 来源 |
|------|------|------|
| 编译通过率 | 生成代码能否 nvcc 编译 | nvcc |
| 正确率 | 所有输入规模上结果正确 | compare .bin 文件 |
| 端到端加速比 | CPU 时间 / GPU 端到端时间（含同步开销） | CUDA Event |
| 纯 Kernel 加速比 | CPU 时间 / 纯 kernel 执行时间 | nsys trace |
| GPU 利用率 | kernel 时间 / 端到端时间 | nsys trace |
| fast_p | 同时正确且加速比 ≥ p 的比例 | 计算得出 |

## nsys 输出的完整信息

使用 `--save-nsys` 时，框架导出 7 种 nsys 报告：

| 报告 | 内容 |
|------|------|
| cuda_gpu_trace | 每个 kernel/memcpy 的逐条记录：开始时间、持续时间、grid/block 配置 |
| cuda_gpu_kern_sum | kernel 按名字汇总：总时间、平均时间、调用次数 |
| cuda_gpu_mem_time_trace | 每次 memcpy/memset 的方向、大小、吞吐量 |
| cuda_gpu_mem_time_sum | 内存操作按类型汇总（H2D/D2H/memset） |
| cuda_gpu_mem_size_sum | 内存操作按数据量汇总 |
| cuda_api_trace | CPU 端每次 CUDA API 调用（cudaMalloc/cudaFree/cudaLaunch 各花多久） |
| cuda_api_sum | CUDA API 按函数名汇总 |

同时生成 `nsys_summary.txt` 可读摘要，包含 GPU 利用率、kernel 逐项分析、内存传输统计、CUDA API 开销分布。

## CLI 参数说明

```bash
# 评测命令
python run.py eval --run <run_name> [选项]

选项：
  --arch sm_89          GPU 架构（sm_89=L20X/4090, sm_80=A100）
  --gpus 2              用几块 GPU 并行评测
  --timeout 180         单任务超时（秒）
  --no-nsys             跳过 nsys profiling（最快）
  --save-nsys           保存完整 nsys CSV 和摘要到 run 目录
  --tasks bellman_ford  只评测指定任务
```

## 新增任务

1. 复制 `tasks/bellman_ford/` 为模板
2. 修改 `task.json`（难度、输入规模、容差）
3. 编写 `prompt_l1/l2/l3.md`
4. 编写 `cpu_reference.cu`（从 .bin 读入、CPU 计时）
5. 编写 `LLM_input.cu`（模板，标记 LLM 填写区域）
6. 编写 `gen_data.py`（生成 .bin 输入 + expected_dist.bin 输出）
7. 编写 `compare.py`（二进制浮点对比）
8. 运行 `gen_data.py` 生成测试数据

## 设计决策

- **独立 .cu 文件**（非 PyTorch 扩展）：更贴近真实 GPGPU 开发，便于 nsys 直接 profile
- **spawn 进程隔离**：每个评测任务在独立子进程中运行，CUDA context 互不干扰
- **二进制 IO**：输入输出用 .bin 文件，避免 50 万个浮点数 printf 的 IO 瓶颈
- **两级计时**：CUDA Event 拿绝对数字，nsys 拿时间分解（kernel/memcpy/空闲占比）
- **NCU 可选**：在 K8s/Docker 容器中不可用（需要 SYS_ADMIN），框架不依赖它
- **LLM_input.cu 模板**：与 cpu_reference.cu 结构完全一致，LLM 只填算法部分，计时由框架控制

## 环境要求

- Python 3.10+
- CUDA Toolkit 12.0+（nvcc）
- NVIDIA GPU
- nsys（推荐，用于 kernel 级分析）
- pip install: numpy pandas pyyaml anthropic openai tqdm