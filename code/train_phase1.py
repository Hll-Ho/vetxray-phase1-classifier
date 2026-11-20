"""W8 Phase-1 训练入口（A′: ConvNeXt-S@512 / B′: Rad-DINO ViT-B/14 + LoRA@518）。

设计约束（工单 W8-T2-TRAIN-PHASE1 绑定规格，逐条对应）：
  1 模型注册：convnext_small / rad_dino_lora；rad_dino_* 刻意不以 vit 开头，
    以绕开 code/train.py:317 的 "vit* 强制 384" 守卫。
  2 head dispatch：名基显式命中，未命中即 fail-fast（禁止落到 model.fc 默认分支）。
  3 transforms：common.dataset.build_transforms 可注入（XrayDataset 构造器接收
    transform），故 convnext 路径直接传参复用，不复制代码；rad_dino 路径用独立
    Normalize（取值优先取 manifest，缺省取本文件 RD_MEAN/RD_STD 卡面常量，
    带 SHA256 锚，禁止凭记忆填值）。两臂各用各自几何策略，但臂内 train/val 一致。
  4 ASL：调用点逐字 AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05,
    disable_torch_grad_focal_loss=True)，reduction 用 .mean() 包一层；
    gamma_pos=0 必须显式传（官方类默认 gamma_pos=1 是陷阱）。
  5 防漂移：RECIPE 常量与 code/train.py 逐字对齐并带行号锚；--recipe-check
    启动时对照 W3 args.json，不一致即退出。
  6 冻结：convnext = freeze-all + 解冻 head；rad_dino = freeze-all + 仅 LoRA
    可训，并打印非零可训参数数（0 即报错）。
  7 探路设施：probe_batch 阶梯 / B-4 反冻结 6.5GB 门逐字沿用；metrics.json 字段名
    与 W1/W3 保持一致。
  8 下载隔离：本文件不含任何联网代码（HF_HUB_OFFLINE 强制为 1），权重获取只走
    code/w8/download_raddino.py。

用法
----
python code/w8/train_phase1.py --model convnext_small --probe \
    --train-file ... --val-file ... --out-root outputs/W8/phase1_probe

本文件不修改 common/utils.py 与 code/train.py。
"""
import os

# 规格 8：训练入口永久离线；下载只允许在 download_raddino.py 中显式临时覆盖
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TORCH_HOME", r"<VETPETS_ROOT>\pretrained\torch")
os.environ.setdefault("HF_HOME", r"<VETPETS_ROOT>\pretrained\hf")

import argparse
import json
import math
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.dataset import (XrayDataset, build_transforms, read_labels,
                            read_name_list, validate_split)
from common.metrics import LOW_FREQ_THRESHOLD, compute_metrics, low_frequency_classes
from common.utils import (IMAGENET_MEAN, IMAGENET_STD, autocast, count_params,
                          get_logger, make_grad_scaler, peak_vram_gb,
                          reset_peak_vram, save_json, seed_all, state_dict_cpu)
from common.viz import plot_per_class_ap, plot_training_curves

# ---------------------------------------------------------------- 规格 1 注册表
W8_MODELS = ("convnext_small", "rad_dino_lora")
DEFAULT_SIZE = {"convnext_small": 512, "rad_dino_lora": 518}

# ------------------------------------------------- 规格 5 配方常量（带行号锚）
# 逐字对齐 code/train.py：epochs L70 / stage1 L71 / patience L72 / batch L73 /
# lr_head L80 / lr_backbone L81 / weight_decay L82 / threshold L86
RECIPE = {
    "epochs": 20,
    "stage1_epochs": 2,
    "patience": 5,
    "batch_size": 16,
    "lr_head": 1e-3,
    "lr_backbone": 1e-4,
    "weight_decay": 1e-4,
    "threshold": 0.5,
}

RECIPE_CHECK_FIELDS = ("epochs", "stage1_epochs", "patience", "batch_size",
                       "lr_head", "lr_backbone", "weight_decay", "threshold")

W3_ARGS_DEFAULT = r"<VETPETS_ROOT>\outputs\W3\C_swa\c2\seed_42\convnext_tiny\args.json"


# ------------------------------------------------------------ 规格 4 ASL 损失
class AsymmetricLoss(nn.Module):
    """ASL 官方实现的等价移植（Ridnik et al., ICCV 2021）。

    签名与默认值与 Alibaba-MIIL/ASL 的 src/loss_functions/losses.py 一致：
        def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                     disable_torch_grad_focal_loss=True)
    正典调用必须显式传 gamma_pos=0（类默认 1 是陷阱）。
    概率裕度只有一个名字一个值：代码名 clip（=0.05），文档名 m，禁止双写。
    官方 forward 返回 sum；本实现返回逐元素损失，调用点用 .mean() 落到
    与 bce/pos_weight 臂一致的 mean 口径（规格 4）。
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8,
                 disable_torch_grad_focal_loss=True):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        # 官方：x_sigmoid = x.sigmoid(); xs_pos = x_sigmoid; xs_neg = 1 - x_sigmoid
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # 非对称概率裕度：负样本 p < m 的部分硬阈值置零（clip ≡ m）
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # 官方：los_pos / los_neg，focusing 权重可选 detach
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                one_sided_w = one_sided_w.detach()
            los_pos = los_pos * one_sided_w
            los_neg = los_neg * one_sided_w

        return -(los_pos + los_neg)


ASL_CANON = dict(gamma_neg=4, gamma_pos=0, clip=0.05,
                 disable_torch_grad_focal_loss=True)


# ------------------------------------------------------------- 规格 2 head 表
def head_module(model_name, model):
    """名基显式命中；未命中即 fail-fast（禁止静默落到 model.fc）。"""
    if model_name.startswith("convnext_"):
        return model.classifier
    if model_name == "rad_dino_lora":
        head = getattr(model, "head", None)
        if not isinstance(head, nn.Linear):
            raise RuntimeError(
                "rad_dino_lora: model.head 不是 nn.Linear（实际 %s）——"
                "检查 timm 是否以 num_classes=0 创建、且 head 已被替换" % type(head))
        return head
    raise ValueError("未知模型名 %r：必须显式命中 head dispatch，禁止默认分支" % model_name)


def head_parameters(model_name, model):
    return list(head_module(model_name, model).parameters())


# ------------------------------------------------------------ 规格 1 构造模型
def build_model_w8(model_name, num_classes, pretrained=True, rd_manifest=None):
    if model_name not in W8_MODELS:
        raise ValueError("model 必须是 %s 之一，得到 %r" % (W8_MODELS, model_name))
    if model_name.startswith("vit"):
        raise ValueError("模型名不得以 vit 开头（会触发 train.py 的 384 守卫语义）")

    if model_name == "convnext_small":
        from torchvision.models import ConvNeXt_Small_Weights, convnext_small
        w = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = convnext_small(weights=w)
        in_f = m.classifier[2].in_features
        m.classifier[2] = nn.Linear(in_f, num_classes)
        return m

    # rad_dino_lora：HF microsoft/rad-dino 的 config.json 是 Dinov2Model
    # （hidden 768 / 12 layers / 12 heads / patch 14 / image 518 / qkv_bias true），
    # 既不能走 timm 的 hf-hub 通道（env 无 transformers），也不能用 peft
    # （env 无 peft 且禁止 pip install）。裁决路线：timm 建
    # vit_base_patch14_dinov2(pretrained=False) + 手工键映射灌入 HF safetensors。
    import timm

    m = timm.create_model("vit_base_patch14_dinov2", pretrained=False,
                          img_size=518, num_classes=0, dynamic_img_size=False)
    if pretrained:
        load_rad_dino_weights(m, rd_weights_path(rd_manifest))

    # timm num_classes=0 时 head 是 Identity，替换为 Linear 供 head dispatch 命中
    m.head = nn.Linear(m.num_features, num_classes)

    # LoRA 恒等断言：lora_B 零初始化，注入不得改变前向（eval + no_grad 逐元素比较）
    torch.manual_seed(0)
    x = torch.randn(2, 3, 518, 518)
    m.eval()
    with torch.no_grad():
        out_before = m(x)
    inject_lora(m, r=8, alpha=16, target="qkv")
    with torch.no_grad():
        out_after = m(x)
    delta = float((out_after - out_before).abs().max())
    if not delta < 1e-6:
        raise RuntimeError("LoRA 注入改变了前向：max|Δ|=%.3e（lora_B 必须为零初始化）"
                           % delta)
    m.train()  # 恢复默认训练态（与 convnext 分支返回状态一致）
    return m


class LoRALinear(nn.Module):
    """最小 LoRA 适配层（等价于 peft 的 r/alpha 设定，无新增依赖）。"""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear 只包装 nn.Linear，得到 %s" % type(base))
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def inject_lora(model, r=8, alpha=16, target="qkv"):
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise RuntimeError("rad_dino_lora: 模型无 .blocks，无法定位 attention 注入点")
    n = 0
    for blk in blocks:
        attn = getattr(blk, "attn", None)
        mod = getattr(attn, target, None) if attn is not None else None
        if isinstance(mod, nn.Linear):
            setattr(attn, target, LoRALinear(mod, r, alpha))
            n += 1
    if n == 0:
        raise RuntimeError("rad_dino_lora: 未找到可注入的 attn.%s，拒绝静默空转" % target)
    return n


# --------------------------------------------- 规格 1b HF safetensors 手工映射
def rd_weights_path(rd_manifest):
    """从 manifest 的 files 段解析本地 .safetensors 路径（manifest 目录 + 相对路径）。"""
    hint = "权重未下载，请先运行 code/w8/download_raddino.py"
    if not rd_manifest:
        raise RuntimeError("rad_dino_lora: 未提供 rd manifest —— " + hint)
    mdir = rd_manifest.get("_manifest_dir")
    if not mdir:
        raise RuntimeError("rad_dino_lora: manifest 缺少所在目录信息（应经 "
                           "load_rd_manifest 载入）—— " + hint)
    cands = sorted(rel for rel in rd_manifest.get("files", {})
                   if rel.endswith(".safetensors"))
    if not cands:
        raise RuntimeError("rad_dino_lora: manifest 的 files 段无 .safetensors 条目 —— "
                           + hint)
    preferred = [c for c in cands if os.path.basename(c) == "model.safetensors"]
    rel = (preferred or cands)[0]
    path = os.path.join(mdir, rel)
    if not os.path.isfile(path):
        raise RuntimeError("rad_dino_lora: 权重文件不存在 %s —— %s" % (path, hint))
    return path


def load_rad_dino_weights(model, safetensors_path):
    """把 HF Dinov2Model 的 safetensors 键映射进 timm vit_base_patch14_dinov2。

    三道强制校验：(a) 键覆盖完备（无缺失/无孤儿）、(b) shape 与 dtype 全检、
    (c) 前向健全性（有限值/非退化方差/量级合理）。任一不满足即 raise。
    """
    from safetensors.torch import load_file

    if not os.path.isfile(safetensors_path):
        raise RuntimeError("rad_dino_lora: 权重文件不存在 %s —— 权重未下载，"
                           "请先运行 code/w8/download_raddino.py" % safetensors_path)

    hf = load_file(safetensors_path)
    hf_keys = set(hf)

    # SwiGLU 检测：HF config 的 use_swiglu_ffn=false，若权重里真出现 fc3 则禁止
    # 静默丢弃，也不能硬套 timm 的两层 MLP。
    fc3 = sorted(k for k in hf_keys if ".mlp.fc3." in k)
    if fc3:
        raise RuntimeError(
            "检测到 fc3/SwiGLU（%d 个键，例如 %s）：当前 timm vit_base_patch14_dinov2 "
            "架构不支持，停止并回报，不得静默丢弃" % (len(fc3), fc3[:3]))

    mapped, consumed, missing = {}, set(), []

    def put(dst, src):
        t = hf.get(src)
        if t is None:
            missing.append(src)
            return
        mapped[dst] = t
        consumed.add(src)

    put("cls_token", "embeddings.cls_token")
    put("pos_embed", "embeddings.position_embeddings")
    put("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight")
    put("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias")

    for i in range(12):
        h = "encoder.layer.%d." % i
        t = "blocks.%d." % i
        put(t + "norm1.weight", h + "norm1.weight")
        put(t + "norm1.bias", h + "norm1.bias")
        for suffix in ("weight", "bias"):
            srcs = [h + "attention.attention.%s.%s" % (n, suffix)
                    for n in ("query", "key", "value")]
            parts = [hf.get(s) for s in srcs]
            if any(p is None for p in parts):
                missing.extend(s for s, p in zip(srcs, parts) if p is None)
            else:
                mapped[t + "attn.qkv." + suffix] = torch.cat(parts, dim=0)
                consumed.update(srcs)
        put(t + "attn.proj.weight", h + "attention.output.dense.weight")
        put(t + "attn.proj.bias", h + "attention.output.dense.bias")
        put(t + "ls1.gamma", h + "layer_scale1.lambda1")
        put(t + "mlp.fc1.weight", h + "mlp.fc1.weight")
        put(t + "mlp.fc1.bias", h + "mlp.fc1.bias")
        put(t + "mlp.fc2.weight", h + "mlp.fc2.weight")
        put(t + "mlp.fc2.bias", h + "mlp.fc2.bias")
        # 注意：HF DinoV2 对 scale1 / scale2 都写作 lambda1（上游命名怪癖，
        # 见 microsoft/rad-dino model.safetensors 实际键），不是 lambda2，勿"顺手修正"
        put(t + "ls2.gamma", h + "layer_scale2.lambda1")
        put(t + "norm2.weight", h + "norm2.weight")
        put(t + "norm2.bias", h + "norm2.bias")

    put("norm.weight", "layernorm.weight")
    put("norm.bias", "layernorm.bias")

    if missing:
        raise RuntimeError("HF 权重缺少下列映射源键（映射未闭合，共 %d 个）：%s"
                           % (len(missing), missing))

    sd = model.state_dict()
    required = {k for k in sd if not k.startswith("head.")}
    absent = sorted(required - set(mapped))
    if absent:
        raise RuntimeError("timm 模型有 %d 个参数未被 HF 权重覆盖：%s"
                           % (len(absent), absent))
    # orphan 豁免集（显式白名单；白名单之外照旧报错，不降级为警告）：
    #   head.*                  —— 分类头，由本脚本自建
    #   embeddings.mask_token   —— MIM 专用 [1,768] F32，特征提取路径不消费
    # 证据：HF 官方 tensor 清单 223 = 222（被消费）+ 1（本条孤儿）
    orphans = sorted(k for k in (hf_keys - consumed)
                     if not (k.startswith("head.") or k == "embeddings.mask_token"))
    if orphans:
        raise RuntimeError("HF 权重有 %d 个键未被消费（可能是前缀/pooler/新结构）：%s"
                           % (len(orphans), orphans))

    bad_shape = [(k, tuple(sd[k].shape), tuple(mapped[k].shape))
                 for k in sorted(mapped) if tuple(sd[k].shape) != tuple(mapped[k].shape)]
    if bad_shape:
        raise RuntimeError("shape 不一致 %d 处（timm vs HF）：%s" % (len(bad_shape), bad_shape))
    bad_dtype = sorted(k for k, v in mapped.items() if v.dtype != torch.float32)
    if bad_dtype:
        raise RuntimeError("下列映射张量不是 float32：%s" % bad_dtype)

    model.load_state_dict(mapped, strict=False)

    bad_param = sorted(n for n, p in model.named_parameters() if p.dtype != torch.float32)
    if bad_param:
        raise RuntimeError("加载后模型参数不是 float32：%s" % bad_param)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 518, 518))
    finite = bool(torch.isfinite(out).all())
    std = float(out.std())
    mean_abs = abs(float(out.mean()))
    if was_training:
        model.train()
    if not (finite and std > 1e-6 and mean_abs <= 10.0):
        raise RuntimeError(
            "前向健全性检查失败：finite=%s std=%.3e (需 >1e-6) "
            "|mean|=%.3e (需 <=10.0)" % (finite, std, mean_abs))
    return {"safetensors": safetensors_path, "n_mapped": len(mapped),
            "out_std": std, "out_mean_abs": mean_abs}


# --------------------------------------------------- 规格 3 rad_dino 变换
# 来源 microsoft/rad-dino/preprocessor_config.json（image_mean/image_std）；
# 该文件 756 字节，SHA256 c537fc995c30e2353f07253899618d60e9eae3d5f82473778602c007c6523b56，
# 2026-11-18 经 hf-mirror 取得（本机 DNS 无法解析 huggingface.co）。
# 禁止使用记忆值 0.449/0.226。
RD_MEAN = (0.5307, 0.5307, 0.5307)
RD_STD = (0.2583, 0.2583, 0.2583)


def build_transforms_rad_dino(size, train, rd_mean=None, rd_std=None, texture=False):
    """Rad-DINO 臂（B′）预处理：严格对齐官方 shortest_edge 518 + center_crop 518。

    val：SmallestMaxSize(size) + CenterCrop(size, size)（官方 eval 流程）。
    train：RandomResizedCrop(size, scale=(0.7,1.0), ratio=(0.85,1.18)) + 原有
    光度/几何增广（ShiftScaleRotate / RandomBrightnessContrast，p=0.3），
    texture=True 时保留 Sharpen/CLAHE。
    归一化取 rd_mean/rd_std，缺省用卡面常量 RD_MEAN/RD_STD（manifest 优先）。
    注：A′（ConvNeXt-S）仍用 W3 的方形 resize 口径，两臂各自内部 train/val 一致，
    不做跨臂几何对齐。
    """
    import albumentations as A

    mean = tuple(rd_mean) if rd_mean is not None else RD_MEAN
    std = tuple(rd_std) if rd_std is not None else RD_STD

    if train:
        ops = [
            A.RandomResizedCrop(size=(size, size), scale=(0.7, 1.0),
                                ratio=(0.85, 1.18), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.10,
                               rotate_limit=10, p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ]
        if texture:
            ops += [
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.9, 1.1), p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            ]
    else:
        ops = [
            A.SmallestMaxSize(max_size=size),
            A.CenterCrop(size, size, p=1.0),
        ]
    ops.append(A.Normalize(mean=mean, std=std))
    return A.Compose(ops)


def load_rd_manifest(path):
    if not os.path.isfile(path):
        raise RuntimeError("Rad-DINO manifest 不存在：%s —— 权重未下载，"
                           "请先运行 code/w8/download_raddino.py" % path)
    with open(path, "r", encoding="utf-8-sig") as f:
        man = json.load(f)
    # 记录 manifest 所在目录，供 rd_weights_path 把 files 相对路径拼成绝对路径
    man["_manifest_dir"] = os.path.dirname(os.path.abspath(path))
    return man


# ------------------------------------------- 规格 7 探路设施（与 train.py 一致）
def make_loaders(train_ds, val_ds, batch_size, workers, prefetch=4):
    persistent = workers > 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True,
                              worker_init_fn=_worker_init, drop_last=False,
                              persistent_workers=persistent,
                              prefetch_factor=(prefetch if workers > 0 else None))
    val_loader = DataLoader(val_ds, batch_size=min(16, batch_size), shuffle=False,
                            num_workers=workers, pin_memory=True,
                            worker_init_fn=_worker_init, drop_last=False,
                            persistent_workers=persistent,
                            prefetch_factor=(prefetch if workers > 0 else None))
    return train_loader, val_loader


def _worker_init(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def probe_batch(model, size, num_classes, base_batch, logger):
    """OOM 阶梯：16 -> 8+acc2 -> 4+acc4（有效 batch 恒 16），逐字沿用 train.py:117。"""
    if base_batch >= 16:
        candidates = [(16, 1), (8, 2), (4, 4)]
    elif base_batch == 8:
        candidates = [(8, 1), (4, 2)]
    else:
        candidates = [(max(1, base_batch), 1)]
    scaler = make_grad_scaler()
    for bs, accum in candidates:
        try:
            torch.cuda.empty_cache()
            x = torch.randn(bs, 3, size, size, device="cuda")
            y = torch.rand(bs, num_classes, device="cuda")
            with autocast():
                out = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), y)
            scaler.scale(loss).backward()
            model.zero_grad(set_to_none=True)
            del x, y, out, loss
            torch.cuda.empty_cache()
            logger.info("batch probe OK: batch=%d grad_accum=%d (effective=%d)",
                        bs, accum, bs * accum)
            return bs, accum
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            logger.warning("batch probe OOM at batch=%d -> next ladder step", bs)
    raise RuntimeError("all batch ladder steps OOM; stop and report (do not change env)")


def _smi_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0].strip())
    except Exception:
        return None


def probe_unfrozen(model, loader, optimizer, scaler, accum, logger, steps=30):
    """B-4：反冻结后实跑 30 步，torch 峰值与 nvidia-smi 峰值并报（逐字沿用 train.py:269）。"""
    reset_peak_vram()
    torch.cuda.empty_cache()
    model.train()
    it = iter(loader)
    n = 0
    smi_max = _smi_used_mb() or 0.0
    try:
        for i in range(steps):
            x, y, _ = next(it)
            if i == steps - 1 and x.size(0) > 8:
                x, y = x[:8], y[:8]
            x = x.to("cuda", non_blocking=True)
            y = y.to("cuda", non_blocking=True)
            with autocast():
                out = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), y)
            scaler.scale(loss / accum).backward()
            if (i + 1) % accum == 0 or i == steps - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            n += 1
            m = _smi_used_mb()
            if m:
                smi_max = max(smi_max, m)
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return None, smi_max, n
    return peak_vram_gb(), smi_max, n


# ------------------------------------------------------------------ 前沿/损失
def make_loss_fn(kind):
    if kind == "bce":
        return lambda out, y, pw=None: torch.nn.functional.binary_cross_entropy_with_logits(
            out.float(), y, pos_weight=pw)
    if kind == "focal":
        def _focal(out, y, pw=None, gamma=2.0):
            ce = torch.nn.functional.binary_cross_entropy_with_logits(
                out.float(), y, pos_weight=pw, reduction="none")
            p = torch.sigmoid(out.float())
            p_t = p * y + (1 - p) * (1 - y)
            return ((1 - p_t) ** gamma * ce).mean()
        return _focal
    if kind == "asl":
        asl = AsymmetricLoss(**ASL_CANON)
        if not (asl.gamma_pos == 0 and asl.gamma_neg == 4 and asl.clip == 0.05):
            raise RuntimeError("ASL 锁参漂移：必须 gamma_neg=4/gamma_pos=0/clip=0.05")
        # 官方 sum -> 我方 mean（规格 4）
        return lambda out, y, pw=None: asl(out.float(), y).mean()
    raise ValueError("loss 必须是 bce/focal/asl 之一")


def train_one_epoch(model, loader, optimizer, scaler, scheduler, accum, logger,
                    loss_fn, pos_weight=None):
    model.train()
    total_loss, seen, n_steps = 0.0, 0, 0
    data_wait, compute = 0.0, 0.0
    optimizer.zero_grad(set_to_none=True)
    n_batches = len(loader)
    t_it = time.time()
    it = iter(loader)
    iterator_sec = time.time() - t_it
    for i in range(n_batches):
        t0 = time.time()
        x, y, _ = next(it)
        data_wait += time.time() - t0
        x = x.to("cuda", non_blocking=True)
        y = y.to("cuda", non_blocking=True)
        if i == 0:
            torch.cuda.synchronize()
        t1 = time.time()
        with autocast():
            out = model(x)
            loss = loss_fn(out, y, pos_weight)
        scaler.scale(loss / accum).backward()
        if (i + 1) % accum == 0 or (i + 1) == n_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            n_steps += 1
        torch.cuda.synchronize()
        compute += time.time() - t1
        total_loss += float(loss.item()) * x.size(0)
        seen += x.size(0)
    if n_steps == 0:
        raise RuntimeError("no optimizer step was performed")
    return {"loss": total_loss / max(1, seen), "steps": n_steps, "images": seen,
            "data_wait": data_wait, "compute": compute, "iterator_sec": iterator_sec}


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, targets = [], []
    for x, y, _ in loader:
        x = x.to("cuda", non_blocking=True)
        with autocast():
            out = model(x)
        probs.append(torch.sigmoid(out.float()).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(targets, axis=0)


# --------------------------------------------------------------- 规格 6 冻结
def freeze_for_stage1(model, model_name):
    for p in model.parameters():
        p.requires_grad = False
    if model_name == "convnext_small":
        for p in head_parameters(model_name, model):
            p.requires_grad = True
    else:  # rad_dino_lora：仅 LoRA 可训
        n_lora = 0
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True
                n_lora += p.numel()
        if n_lora == 0:
            raise RuntimeError("rad_dino_lora: 无 LoRA 参数可训（整模型被冻结成摆设）")


def unfreeze_stage2(model, model_name, lr_backbone, lr_head, weight_decay):
    if model_name == "convnext_small":
        for p in model.parameters():
            p.requires_grad = True
        head_ids = {id(p) for p in head_parameters(model_name, model)}
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        head = [p for p in model.parameters() if id(p) in head_ids]
        groups = [{"params": backbone, "lr": lr_backbone, "name": "backbone"},
                  {"params": head, "lr": lr_head, "name": "head"}]
    else:
        lora, head = [], []
        head_ids = {id(p) for p in head_parameters(model_name, model)}
        for n, p in model.named_parameters():
            if id(p) in head_ids:
                p.requires_grad = True
                head.append(p)
            elif "lora_" in n:
                p.requires_grad = True
                lora.append(p)
            else:
                p.requires_grad = False
        groups = [{"params": lora, "lr": lr_backbone, "name": "lora"},
                  {"params": head, "lr": lr_head, "name": "head"}]
    trainable = sum(p.numel() for g in groups for p in g["params"])
    if trainable == 0:
        raise RuntimeError("stage2: 可训参数为 0，拒绝启动")
    return torch.optim.AdamW(groups, weight_decay=weight_decay), trainable


def report_trainable(logger, model_name, model):
    total, trainable = count_params(model)
    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    logger.info("params: total=%.1fM trainable=%.1fM (lora=%.1fM)",
                total / 1e6, trainable / 1e6, n_lora / 1e6)
    if trainable <= 0:
        raise RuntimeError("可训参数为 0：冻结配置错误")


# ------------------------------------------------------------------- 规格 5
def recipe_check(path, model_name, size, logger, labels=None):
    with open(path, "r", encoding="utf-8-sig") as f:
        w3 = json.load(f)
    a = w3.get("args", {})
    bad = []
    for k in RECIPE_CHECK_FIELDS:
        if k in a and a[k] != RECIPE[k]:
            bad.append("%s: W3=%r vs 本入口=%r" % (k, a[k], RECIPE[k]))
    if model_name == "convnext_small" and a.get("size") not in (None, size):
        bad.append("size: W3=%r vs 本入口=%r" % (a.get("size"), size))
    # labels 只比文件名：盘符跨机器会变（W3=F: / 本机=E:），文件名是数据宇宙的实际锚
    if labels and a.get("labels"):
        if os.path.basename(a["labels"]) != os.path.basename(labels):
            bad.append("labels: W3=%r vs 本入口=%r"
                       % (os.path.basename(a["labels"]), os.path.basename(labels)))
    if bad:
        logger.error("recipe check FAILED against %s", path)
        for b in bad:
            logger.error("  %s", b)
        raise SystemExit(3)
    w3_seed = a.get("seed")
    logger.info("recipe check PASS: %d 字段与 W3 一致；W3 seed=%s", len(RECIPE_CHECK_FIELDS), w3_seed)
    return w3_seed


# ------------------------------------------------------------------ 命令行
def parse_args():
    p = argparse.ArgumentParser("W8 phase-1 trainer (A'/B' arms)")
    p.add_argument("--model", required=True, choices=list(W8_MODELS))
    p.add_argument("--data-dir", default=r"<VETPETS_ROOT>\data")
    p.add_argument("--images-dir", default=None)
    p.add_argument("--labels", default=None)
    p.add_argument("--train-file", required=True)
    p.add_argument("--val-file", required=True)
    p.add_argument("--out-root", default=r"<VETPETS_ROOT>\outputs\W8\phase1")
    p.add_argument("--tag", default="", help="子目录名（探路用 probe 等）")
    p.add_argument("--epochs", type=int, default=RECIPE["epochs"])
    p.add_argument("--stage1-epochs", type=int, default=RECIPE["stage1_epochs"])
    p.add_argument("--patience", type=int, default=RECIPE["patience"])
    p.add_argument("--batch-size", type=int, default=RECIPE["batch_size"])
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--seed", type=int, default=None,
                   help="缺省时取 W3 args.json 的 seed（探路对齐）")
    p.add_argument("--lr-head", type=float, default=RECIPE["lr_head"])
    p.add_argument("--lr-backbone", type=float, default=RECIPE["lr_backbone"])
    p.add_argument("--weight-decay", type=float, default=RECIPE["weight_decay"])
    p.add_argument("--pos-weight", choices=["none", "auto"], default="none")
    p.add_argument("--texture-aug", action="store_true")
    p.add_argument("--threshold", type=float, default=RECIPE["threshold"])
    p.add_argument("--loss", choices=["bce", "focal", "asl"], default="bce")
    p.add_argument("--recipe-check", default=W3_ARGS_DEFAULT,
                   help="对照 W3 args.json；传空串跳过（不推荐）")
    p.add_argument("--probe", action="store_true",
                   help="探路模式：产物标 NON-CONSUMABLE，不得进任何门或主张")
    p.add_argument("--rd-manifest", default=r"<VETPETS_ROOT>\pretrained\raddino\manifest_raddino.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.probe and args.epochs == RECIPE["epochs"]:
        args.epochs = 4  # 探路只需覆盖 stage1 + 反冻结首 epoch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA required")

    size = args.size or DEFAULT_SIZE[args.model]
    images_dir = args.images_dir or os.path.join(args.data_dir, "images")
    labels_path = args.labels or os.path.join(args.data_dir, "labels.csv")
    out_dir = os.path.join(args.out_root, args.tag or "", args.model)
    # 同 tag 禁重跑：目录已存在即秒级失败，旧目录原样保留（失败现场也是证据）。
    # 不提供 --force；确认为空目录时可手动删除后重跑。
    if os.path.isdir(out_dir):
        raise RuntimeError(
            "输出目录已存在: %s —— 同 tag 禁重跑，换 --tag（如 probe2）；"
            "旧目录原样保留，失败现场也是证据" % out_dir)
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    logger = get_logger(os.path.join(out_dir, "log.txt"))
    logger.info("=== W8 phase1 train: model=%s size=%d probe=%s ===",
                args.model, size, args.probe)

    w3_seed = None
    if args.recipe_check:
        w3_seed = recipe_check(args.recipe_check, args.model, size, logger,
                               labels=args.labels)
    if args.seed is None:
        if w3_seed is None:
            raise SystemExit("--seed 未给出且无 W3 参照：拒绝用不确定 seed 起跑")
        args.seed = int(w3_seed)
        logger.info("seed 对齐 W3：%d", args.seed)
    seed_all(args.seed)

    df, classes = read_labels(labels_path)
    train_names = read_name_list(args.train_file)
    val_names = read_name_list(args.val_file)
    logger.info("classes(%d frozen order): %s", len(classes), classes)
    ok, _ = validate_split(df, classes, train_names, val_names,
                           out_path=os.path.join(out_dir, "split_check.txt"),
                           split_files=[args.train_file, args.val_file], logger=logger)
    if not ok:
        raise SystemExit(2)

    man = None
    if args.model == "convnext_small":
        # 规格 3：可注入即传参，不复制 build_transforms
        tr_tf = build_transforms(size, train=True, texture=args.texture_aug)
        va_tf = build_transforms(size, train=False)
    else:
        man = load_rd_manifest(args.rd_manifest)
        norm = man.get("normalize") or {}
        if norm.get("mean") and norm.get("std"):
            rd_mean, rd_std = norm["mean"], norm["std"]
            logger.info("rad-dino normalize from manifest: mean=%s std=%s (source=%s)",
                        rd_mean, rd_std, norm.get("source", "n/a"))
        else:
            rd_mean, rd_std = list(RD_MEAN), list(RD_STD)
            logger.info("rad-dino normalize from module constant (card, SHA c537fc…): "
                        "mean=%s std=%s", rd_mean, rd_std)
        tr_tf = build_transforms_rad_dino(size, True, rd_mean, rd_std, args.texture_aug)
        va_tf = build_transforms_rad_dino(size, False, rd_mean, rd_std)

    train_ds = XrayDataset(images_dir, train_names, classes, df, tr_tf)
    val_ds = XrayDataset(images_dir, val_names, classes, df, va_tf)
    logger.info("data: train=%d val=%d images_dir=%s", len(train_ds), len(val_ds), images_dir)

    pos_weight = None
    if args.pos_weight == "auto":
        pos = df[df["filename"].isin(set(train_names))][classes].sum().to_numpy()
        neg = len(train_names) - pos
        pw = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
        logger.info("pos_weight(auto): %s", [round(float(v), 2) for v in pw])

    model = build_model_w8(args.model, len(classes), pretrained=True,
                           rd_manifest=(man if args.model == "rad_dino_lora" else None)
                           ).to(device)
    loss_fn = make_loss_fn(args.loss)
    logger.info("loss=%s", args.loss)

    batch_size, accum = probe_batch(model, size, len(classes), args.batch_size, logger)
    train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size,
                                            args.workers, args.prefetch_factor)

    stage1_epochs = min(args.stage1_epochs, max(1, args.epochs - 1))
    stage1_epochs = max(1, stage1_epochs)
    freeze_for_stage1(model, args.model)
    report_trainable(logger, args.model, model)
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad],
          "lr": args.lr_head, "name": "head"}],
        weight_decay=args.weight_decay)
    scaler = make_grad_scaler()
    scheduler = None
    stage2_started = False
    logger.info("stage1 (frozen backbone) epochs=%d lr=%.0e", stage1_epochs, args.lr_head)

    records, oom_events = [], []
    best_map, best_epoch = -1.0, 0
    epoch = 1
    unfrozen_probe_vram = None
    unfrozen_probe_info = {}
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    def rebuild_after_oom():
        nonlocal train_loader, val_loader
        train_loader, val_loader = make_loaders(train_ds, val_ds, batch_size,
                                                args.workers, args.prefetch_factor)
        logger.warning("degraded to batch=%d grad_accum=%d", batch_size, accum)

    while epoch <= args.epochs:
        if epoch == stage1_epochs + 1 and not stage2_started:
            optimizer, trainable2 = unfreeze_stage2(model, args.model, args.lr_backbone,
                                                    args.lr_head, args.weight_decay)
            logger.info("stage2 trainable=%.1fM (backbone/lora lr=%.0e head lr=%.0e)",
                        trainable2 / 1e6, args.lr_backbone, args.lr_head)
            steps_per_epoch = math.ceil(len(train_loader) / accum)
            total_steps = max(1, steps_per_epoch * (args.epochs - stage1_epochs))
            warmup_steps = steps_per_epoch

            def lr_lambda(step, warmup=warmup_steps, total=total_steps):
                if step < warmup:
                    return float(step + 1) / float(max(1, warmup))
                p = min(1.0, (step - warmup) / float(max(1, total - warmup)))
                return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * p))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            stage2_started = True

            while True:
                pv, smi_mb, pn = probe_unfrozen(model, train_loader, optimizer, scaler,
                                                accum, logger)
                peak_gb = max(pv or 0.0, (smi_mb or 0.0) / 1024.0)
                if pv is not None and peak_gb < 6.5:
                    unfrozen_probe_vram = float(peak_gb)
                    unfrozen_probe_info.update({
                        "peak_gb": float(peak_gb), "torch_peak_gb": float(pv),
                        "smi_max_mb": float(smi_mb or 0.0), "steps": pn,
                        "batch_size": batch_size, "grad_accum": accum, "verdict": "keep"})
                    logger.info("B-4 probe OK: %d steps peak %.2fGB < 6.5GB -> keep bs=%d",
                                pn, peak_gb, batch_size)
                    break
                logger.warning("B-4 probe %s at bs=%d accum=%d peak %.2fGB",
                               "OOM" if pv is None else ">=6.5GB", batch_size, accum, peak_gb)
                if batch_size > 4:
                    batch_size //= 2
                    accum *= 2
                    rebuild_after_oom()
                    continue
                raise RuntimeError("B-4: batch floor (4) still OOM/high; stop and report")

        reset_peak_vram()
        t0 = time.time()
        info = train_one_epoch(model, train_loader, optimizer, scaler, scheduler, accum,
                               logger, loss_fn, pos_weight)
        t_train = time.time() - t0
        probs, targets = evaluate(model, val_loader)
        epoch_sec = time.time() - t0
        vram = peak_vram_gb()
        met = compute_metrics(targets, probs, classes, threshold=args.threshold)
        val_map = met["macro_map"]
        best_now = (not np.isnan(val_map)) and val_map > best_map
        if best_now:
            best_map, best_epoch = float(val_map), epoch
            torch.save({"model": state_dict_cpu(model), "classes": classes,
                        "model_name": args.model, "epoch": epoch,
                        "val_map": best_map, "consumable": not args.probe},
                       os.path.join(out_dir, "best.pth"))
        torch.save({"model": state_dict_cpu(model), "classes": classes,
                    "model_name": args.model, "epoch": epoch,
                    "val_map": float(val_map) if not np.isnan(val_map) else None,
                    "consumable": not args.probe}, os.path.join(out_dir, "last.pth"))

        rec = {"epoch": epoch,
               "stage": "head_warmup" if epoch <= stage1_epochs else "finetune",
               "train_loss": float(info["loss"]),
               "val_map": float(val_map) if not np.isnan(val_map) else None,
               "val_f1": float(met["macro_f1"]) if not np.isnan(met["macro_f1"]) else None,
               "macro_map_note": met["macro_map_note"],
               "n_evaluable": met["n_evaluable"],
               "per_class_ap": {k: (None if np.isnan(v) else float(v))
                                for k, v in met["per_class_ap"].items()},
               "per_class_f1": met["per_class_f1"],
               "n_pos_val": met["n_pos"],
               "optimizer_steps": info["steps"],
               "epoch_sec": float(epoch_sec), "train_sec": float(t_train),
               "data_wait_sec": float(info["data_wait"]),
               "gpu_compute_sec": float(info["compute"]),
               "throughput_img_s": float(info["images"]) / max(t_train, 1e-9),
               "train_images": int(info["images"]),
               "backbone_frozen": bool(epoch <= stage1_epochs),
               "unfrozen_probe_vram_gb": unfrozen_probe_vram,
               "trainable_params": sum(p.numel() for p in model.parameters()
                                       if p.requires_grad),
               "vram_peak_gb": float(vram),
               "batch_size": batch_size, "grad_accum": accum,
               "consumable": not args.probe}
        records.append(rec)
        logger.info("epoch %d [%s] loss=%.4f val_mAP=%.4f val_F1=%.4f time=%.1fs "
                    "vram=%.2fGB%s", epoch, rec["stage"], rec["train_loss"],
                    -1 if rec["val_map"] is None else rec["val_map"],
                    -1 if rec["val_f1"] is None else rec["val_f1"],
                    epoch_sec, vram, " *best" if best_now else "")
        logger.info("  %s", met["macro_map_note"])

        save_json({"model": args.model, "classes": classes, "size": size,
                   "consumable": not args.probe, "purpose": "probe" if args.probe else "main",
                   "loss": args.loss, "seed": args.seed, "args": vars(args),
                   "batch_size": batch_size, "grad_accum": accum,
                   "oom_events": oom_events,
                   "unfrozen_probe": unfrozen_probe_info,
                   "epochs": records, "best_epoch": best_epoch,
                   "best_val_map": best_map if best_epoch else None},
                  os.path.join(out_dir, "metrics.json"))

        writer.add_scalar("loss/train", rec["train_loss"], epoch)
        if rec["val_map"] is not None:
            writer.add_scalar("map/val_macro", rec["val_map"], epoch)
        writer.flush()

        if epoch - best_epoch >= args.patience:
            logger.info("early stop: no val mAP improvement for %d epochs", args.patience)
            break
        epoch += 1

    if records:
        plot_training_curves(records, best_epoch,
                             os.path.join(figs_dir, "training_curve.png"))
        plot_per_class_ap(records[best_epoch - 1]["per_class_ap"], classes,
                          os.path.join(figs_dir, "per_class_ap.png"),
                          title=f"Per-class AP at best epoch {best_epoch}")
    save_json({"args": vars(args), "size": size, "consumable": not args.probe,
               "batch_size": batch_size, "grad_accum": accum,
               "oom_events": oom_events, "best_epoch": best_epoch,
               "best_val_map": best_map if best_epoch else None,
               "unfrozen_probe": unfrozen_probe_info},
              os.path.join(out_dir, "args.json"))
    writer.close()
    logger.info("done. best_epoch=%d best_val_map=%s consumable=%s",
                best_epoch, f"{best_map:.4f}" if best_epoch else "n/a", not args.probe)


if __name__ == "__main__":
    main()
