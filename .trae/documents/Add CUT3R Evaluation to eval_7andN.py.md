# 目标
在 `eval_old/eval_7andN.py` 中新增对 CUT3R 的评估支持，严格对齐 `CUT3R/demo.py` 的调用规范、参数顺序与类型，并将推理输出适配到现有评估流水线（点云对齐与指标计算）。

# 关键改动
## 新增模型分支：CUT3R
- 在 `--model_name` 的分支中加入 `CUT3R`：
  - 复制并引入 CUT3R 的必要导入：
    - `from CUT3R.add_ckpt_path import add_path_to_dust3r`
    - `from src.dust3r.model import ARCroco3DStereo`
    - `from src.dust3r.inference import inference`
  - 新增 CLI 参数：`--cut3r_model_path`（默认 `src/cut3r_512_dpt_4_64.pth`）
  - 初始化：
    - 调用 `add_path_to_dust3r(args.cut3r_model_path)`
    - `model = ARCroco3DStereo.from_pretrained(args.cut3r_model_path)`
    - 设备为 `torch.device(args.device)`；强制 bf16：`model = model.to(device).to(torch.bfloat16).eval()`（若用户坚持完全一致，可保留 fp32；默认按现评估环境 bf16）

## 视图准备：按 CUT3R demo 规范
- 从当前数据集批次 `batch` 提取每视图的原始图像路径：`view['instance']`（已在 `eval_old/data.py` 设置为 `impath`）
- 复刻 demo 的 `prepare_input` 逻辑为本文件中的辅助函数 `cut3r_prepare_input(img_paths, size, revisit=1, update=True)`：
  - 使用 `src.dust3r.utils.image.load_images(img_paths, size=args.size)`
  - 构造每个视图 dict：`img`、`ray_map`（填充 NaN）、`true_shape`、`idx`、`instance`、`camera_pose`（单位矩阵）、`img_mask`、`ray_mask`、`update`、`reset`（与 demo 完全一致）
  - 支持 `revisit>1` 的复制与 `update` 标记
- 参数顺序与类型严格匹配 demo 实现

## 推理与结果适配
- 调用：`outputs, state_args = inference(views, model, device)`（参数顺序与 demo 一致）
- 复刻 demo 的输出后处理，用 `prepare_output` 同步方式（或在本文件内实现等价逻辑）：
  - 计算每视图 `pts3d_in_other_view`（世界坐标）与 `conf`（置信度）
  - 返回结构转为现评估需要的 `preds` 列表，每元素包含：
    - `pts3d_in_other_view`: Tensor[B,H,W,3]
    - `conf`: Tensor[B,H,W]（若无，使用与 demo 相同的 `conf_self/conf`；否则退化为 ones）
    - 可选：`camera_pose`（若后续对齐需要）
- 将 `preds` 与当前 `criterion.get_all_pts3d_t` 流水线打通，复用现有 Umeyama+ICP 对齐与指标计算与日志输出。

## 参数与错误处理
- 参数检查：
  - 确保每个视图均可解析 `instance` 路径（若缺失，记录错误并跳过该样本）
  - 验证 `img_paths` 非空且数量≥3（与 demo 保持一致）
  - 设备可用性检查：CUDA 不可用时退回 CPU，并提示
- 异常处理：
  - 对推理失败场景捕获异常，打印友好信息并继续后续样本处理
  - 在 `prepare_input` 期间若加载失败，输出具体文件路径与原因

## 输出与格式
- 维持与 `eval_7andN.py` 一致的 FPS 统计、逐场景日志与最终聚合日志（`logs_all.txt`）结构不变
- 指标计算：继续使用当前 `accuracy/completion/NC` 的实现，不调整接口

# 集成步骤
1. 在 `get_args_parser()` 中新增 `--cut3r_model_path` 参数
2. 在头部 `sys.path` 增加 CUT3R 模块路径与 ckpt 相对路径适配（与 demo 同步）
3. 新增 `CUT3R` 分支：完成 `add_ckpt_path`、模型加载与设备设置
4. 实现 `cut3r_prepare_input()` 辅助函数，严格复刻 demo 的输入构造
5. 在主要循环中，当 `model_name == CUT3R`：
   - 生成 `img_paths`（从 `batch[i]['instance']`）
   - 构造 CUT3R 视图并调用 `inference`
   - 适配输出到当前评估 `preds` 列表格式
   - 复用现评估流程进行对齐与指标计算
6. 在异常分支中打印明确的错误信息并继续

# 质量保证
- 参数类型与边界验证：
  - `img_paths` 列表字符串性验证；`size` 为整数；`device` 为合法字符串
- 错误处理与日志：
  - 与 demo 一致的容错打印，且对每视图缺失路径进行说明
- 性能与稳定性：
  - 优先使用 bf16（符合现评估环境），视图准备按批次运行，避免重复开销

# 验证
- 在 NRGBD 与 7-scenes 选取一个样本，运行 CUT3R 分支，生成 `logs.txt` 与 `logs_all.txt`
- 检查 `Idx: scene_id, FPS` 行是否存在，指标与对齐是否正常
- 与 demo 的小批量图像测试结果比对，确认接口一致性
