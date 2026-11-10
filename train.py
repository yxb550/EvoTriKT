import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data_provider.paper_aligned_kt import (
    PaperAlignedKTDataset as Assist2009KTDataset,
    load_paper_aligned_metadata as load_assist2009_metadata,
)
from models import EvoTriKT


def parse_args():
    parser = argparse.ArgumentParser(description="Train the EvoTriKT three-view model")
    parser.add_argument("--config", default=str(Path(PROJECT_ROOT) / "scripts" / "train_config.json"))
    parser.add_argument("--data_root", type=str, default="dataset/assist2009")
    parser.add_argument("--save_dir", type=str, default="Results/EvoTriKT/assist2009")
    parser.add_argument("--processed_tag", type=str, default="")
    parser.add_argument("--prompt_embedding_dir", type=str, default=None)
    parser.add_argument("--strict_trimodal", action="store_true")
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--prompt_dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aux_loss_weight", type=float, default=0.0)
    parser.add_argument("--graph_aux_weight", type=float, default=0.0)
    parser.add_argument("--prompt_aux_weight", type=float, default=0.0)
    parser.add_argument("--view_consistency_teacher", choices=["final", "sequence"], default="sequence")
    parser.add_argument("--view_consistency_weight", type=float, default=0.0)
    parser.add_argument("--gate_balance_weight", type=float, default=0.0)
    parser.add_argument("--evidence_balance_weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--sample_percent", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--device_ids", type=str, default=None)
    parser.add_argument("--seq_layers", type=int, default=1)
    parser.add_argument("--prompt_layers", type=int, default=1)
    parser.add_argument("--transformer_heads", type=int, default=4)
    parser.add_argument("--alignment_heads", type=int, default=4)
    parser.add_argument("--semantic_tokens", type=int, default=16)
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--ema_decay", type=float, default=0.0)
    parser.add_argument("--ema_start_epoch", type=int, default=1)
    parser.add_argument("--graph_adjacency_path", type=str, default=None)
    parser.add_argument("--disable_state_forgetting", action="store_true")
    parser.add_argument(
        "--semantic_fusion_mode",
        choices=["legacy", "student_gated"],
        default="legacy",
    )
    parser.add_argument("--semantic_modality_dropout", type=float, default=0.2)
    parser.add_argument("--question_dropout", type=float, default=0.0)
    parser.add_argument("--folibi_alpha", type=float, default=0.0)
    parser.add_argument("--use_response_interaction", action="store_true")
    parser.add_argument("--response_interaction_scale", type=float, default=0.25)
    parser.add_argument("--response_interaction_mode", choices=["concept", "question_concept"], default="concept")
    parser.add_argument("--response_interaction_dropout", type=float, default=0.0)
    parser.add_argument("--use_rich_graph_evidence", action="store_true")
    parser.add_argument("--target_residual_scale", type=float, default=0.5)
    parser.add_argument("--use_target_denoised_pooling", action="store_true")
    parser.add_argument("--use_student_semantic_adapter", action="store_true")
    parser.add_argument("--student_semantic_rank", type=int, default=64)
    parser.add_argument("--student_semantic_dropout", type=float, default=None)
    parser.add_argument("--student_semantic_max_scale", type=float, default=0.25)
    parser.add_argument("--student_semantic_gate_bias", type=float, default=-2.0)
    parser.add_argument("--student_semantic_gate_reg", type=float, default=0.0)
    parser.add_argument("--student_semantic_delta_ratio_reg", type=float, default=0.0)
    parser.add_argument(
        "--student_semantic_adapter_mode",
        choices=["legacy", "conservative"],
        default="legacy",
    )
    parser.add_argument("--use_semantic_advantage_calibration", action="store_true")
    parser.add_argument("--semantic_advantage_max_delta", type=float, default=1.0)
    parser.add_argument("--semantic_advantage_lambda_reg", type=float, default=0.0)
    parser.add_argument("--use_view_residual_calibration", action="store_true")
    parser.add_argument("--view_residual_max_delta", type=float, default=0.6)
    parser.add_argument("--view_residual_gate_reg", type=float, default=0.0)
    parser.add_argument("--view_residual_correction_reg", type=float, default=0.0)
    parser.add_argument("--use_mindts_semantic_readout", action="store_true")
    parser.add_argument("--mindts_semantic_aux_weight", type=float, default=0.0)
    parser.add_argument("--validation_only", action="store_true")
    preview, _ = parser.parse_known_args()
    if preview.config:
        settings = json.loads(Path(preview.config).read_text(encoding="utf-8"))
        supported = {action.dest for action in parser._actions}
        unknown = set(settings) - supported
        if unknown:
            parser.error(f"Unknown configuration keys: {sorted(unknown)}")
        parser.set_defaults(**settings)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def get_prompt_embeddings(batch):
    return batch["prompt_embeddings"] if "prompt_embeddings" in batch else None


def get_history_prompt_embeddings(batch):
    return batch["history_prompt_embeddings"] if "history_prompt_embeddings" in batch else None


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def save_model_state(model, path):
    torch.save(unwrap_model(model).state_dict(), path)


def load_model_state(model, path, device):
    state = torch.load(path, map_location=device)
    target = unwrap_model(model)
    if all(key.startswith("module.") for key in state.keys()):
        state = {key[len("module."):]: value for key, value in state.items()}
    target.load_state_dict(state)


def parse_device_ids(value):
    if value is None or str(value).strip() == "":
        return None
    return [int(item.strip()) for item in str(value).split(",") if item.strip() != ""]


def get_multi_concept_kwargs(model, batch):
    model = unwrap_model(model)
    if not getattr(model, "supports_multi_concept", False):
        return {}
    return {
        key: batch[key]
        for key in (
            "concept_set_ids",
            "concept_set_mask",
            "next_concept_set_ids",
            "next_concept_set_mask",
        )
        if key in batch
    }




@torch.no_grad()
def init_ema(model):
    model = unwrap_model(model)
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad and param.dtype.is_floating_point
    }


@torch.no_grad()
def update_ema(model, ema_state, decay: float):
    model = unwrap_model(model)
    for name, param in model.named_parameters():
        if name in ema_state:
            ema_state[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)


@torch.no_grad()
def copy_ema_to_model(model, ema_state):
    model = unwrap_model(model)
    backup = {}
    for name, param in model.named_parameters():
        if name in ema_state:
            backup[name] = param.detach().clone()
            param.copy_(ema_state[name])
    return backup


@torch.no_grad()
def restore_model_params(model, backup):
    model = unwrap_model(model)
    for name, param in model.named_parameters():
        if name in backup:
            param.copy_(backup[name])


def evaluate_maybe_ema(model, loader, device, ema_state=None):
    if ema_state is None:
        return evaluate(model, loader, device)
    backup = copy_ema_to_model(model, ema_state)
    try:
        return evaluate(model, loader, device)
    finally:
        restore_model_params(model, backup)


def expected_calibration_error(labels, probs, bins=10):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    ece = 0.0
    for i in range(bins):
        lower = i / bins
        upper = (i + 1) / bins
        mask = (probs > lower) & (probs <= upper)
        if not np.any(mask):
            continue
        conf = np.mean(probs[mask])
        acc = np.mean(labels[mask] == (probs[mask] >= 0.5))
        ece += np.mean(mask) * abs(acc - conf)
    return float(ece)


def compute_metrics(labels, probs):
    labels = np.asarray(labels).astype(np.float32)
    probs = np.asarray(probs).astype(np.float32)
    preds = (probs >= 0.5).astype(np.float32)
    eps = 1e-7
    out = {
        "acc": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "rmse": float(np.sqrt(mean_squared_error(labels, probs))),
        "nll": float(-(labels * np.log(probs + eps) + (1 - labels) * np.log(1 - probs + eps)).mean()),
        "brier": float(np.mean((probs - labels) ** 2)),
        "ece": expected_calibration_error(labels, probs),
    }
    try:
        out["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["auc"] = float("nan")
    return out


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_probs = []
    all_base_probs = []
    all_semantic_probs = []
    total_loss = 0.0
    total_count = 0
    conflict = []
    graph_conf = []
    gate_mean = []
    student_semantic_gate = []
    student_semantic_gate_std = []
    student_semantic_delta_norm = []
    student_semantic_base_norm = []
    student_semantic_delta_base_ratio = []
    student_semantic_scale = []
    semantic_advantage_lambda = []
    semantic_correction_mean = []
    semantic_correction_abs = []
    mindts_semantic_gate = []
    mindts_semantic_mask = []
    mindts_semantic_aux_loss = []
    retrieval_density = []
    real_time_available = []
    retrieval_mean_age = []
    rasch_mastery = []
    rasch_difficulty = []
    rasch_discrimination = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(
            question_ids=batch["question_ids"],
            concept_ids=batch["concept_ids"],
            responses=batch["responses"],
            delta_t=batch["delta_t"],
            retention_age=batch.get("retention_age"),
            difficulty=batch["difficulty"],
            padding_mask=batch["padding_mask"],
            next_question_ids=batch["next_question_ids"],
            next_concept_ids=batch["next_concept_ids"],
            prompt_embeddings=get_prompt_embeddings(batch),
            history_prompt_embeddings=get_history_prompt_embeddings(batch),
            student_context_features=batch.get("student_context_features"),
            **get_multi_concept_kwargs(model, batch),
        )
        loss = F.binary_cross_entropy_with_logits(out["logits"], batch["labels"], reduction="sum")
        total_loss += float(loss.detach().cpu())
        total_count += batch["labels"].numel()
        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())
        all_probs.extend(out["probs"].detach().cpu().numpy().tolist())
        all_base_probs.extend(out["base_probs"].detach().cpu().numpy().tolist())
        all_semantic_probs.extend(out["semantic_probs"].detach().cpu().numpy().tolist())
        conflict.extend(out["view_conflict"].detach().cpu().numpy().tolist())
        graph_conf.extend(out["graph_confidence"].detach().cpu().numpy().tolist())
        gate_mean.extend(out["temporal_structural_gate"].mean(dim=1).detach().cpu().numpy().tolist())
        student_semantic_gate.extend(out["student_semantic_gate"].detach().cpu().numpy().tolist())
        student_semantic_gate_std.extend(out["student_semantic_gate_std"].detach().cpu().numpy().tolist())
        student_semantic_delta_norm.extend(out["student_semantic_delta_norm"].detach().cpu().numpy().tolist())
        student_semantic_base_norm.extend(out["student_semantic_base_norm"].detach().cpu().numpy().tolist())
        student_semantic_delta_base_ratio.extend(out["student_semantic_delta_base_ratio"].detach().cpu().numpy().tolist())
        student_semantic_scale.extend(out["student_semantic_scale"].detach().cpu().numpy().tolist())
        semantic_advantage_lambda.extend(out["semantic_advantage_lambda"].detach().cpu().numpy().tolist())
        semantic_correction_mean.extend(out["semantic_correction_mean"].detach().cpu().numpy().tolist())
        semantic_correction_abs.extend(out["semantic_correction_abs"].detach().cpu().numpy().tolist())
        mindts_semantic_gate.extend(out["mindts_semantic_gate"].detach().cpu().numpy().tolist())
        mindts_semantic_mask.extend(out["mindts_semantic_mask"].detach().cpu().numpy().tolist())
        mindts_semantic_aux_loss.extend(out["mindts_semantic_aux_loss"].detach().cpu().numpy().tolist())
        if "retrieval_density" in out:
            retrieval_density.extend(out["retrieval_density"].detach().cpu().numpy().tolist())
        if "real_time_available" in out:
            real_time_available.extend(out["real_time_available"].float().detach().cpu().numpy().tolist())
        if "retrieval_mean_age" in out:
            retrieval_mean_age.extend(out["retrieval_mean_age"].detach().cpu().numpy().tolist())
        if "rasch_mastery" in out:
            rasch_mastery.extend(out["rasch_mastery"].detach().cpu().numpy().tolist())
        if "rasch_difficulty" in out:
            rasch_difficulty.extend(out["rasch_difficulty"].detach().cpu().numpy().tolist())
        if "rasch_discrimination" in out:
            rasch_discrimination.extend(out["rasch_discrimination"].detach().cpu().numpy().tolist())
    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / max(total_count, 1)
    metrics["conflict_mean"] = float(np.mean(conflict)) if conflict else 0.0
    metrics["graph_confidence_mean"] = float(np.mean(graph_conf)) if graph_conf else 0.0
    metrics["gate_mean"] = float(np.mean(gate_mean)) if gate_mean else 0.0
    metrics["student_semantic_gate_mean"] = float(np.mean(student_semantic_gate)) if student_semantic_gate else 0.0
    metrics["student_semantic_gate_std"] = float(np.mean(student_semantic_gate_std)) if student_semantic_gate_std else 0.0
    metrics["student_semantic_delta_norm"] = float(np.mean(student_semantic_delta_norm)) if student_semantic_delta_norm else 0.0
    metrics["student_semantic_base_norm"] = float(np.mean(student_semantic_base_norm)) if student_semantic_base_norm else 0.0
    metrics["student_semantic_delta_base_ratio"] = (
        float(np.mean(student_semantic_delta_base_ratio)) if student_semantic_delta_base_ratio else 0.0
    )
    metrics["student_semantic_scale"] = float(np.mean(student_semantic_scale)) if student_semantic_scale else 0.0
    metrics["semantic_advantage_lambda"] = float(np.mean(semantic_advantage_lambda)) if semantic_advantage_lambda else 0.0
    metrics["semantic_correction_mean"] = float(np.mean(semantic_correction_mean)) if semantic_correction_mean else 0.0
    metrics["semantic_correction_abs"] = float(np.mean(semantic_correction_abs)) if semantic_correction_abs else 0.0
    metrics["mindts_semantic_gate_mean"] = float(np.mean(mindts_semantic_gate)) if mindts_semantic_gate else 0.0
    metrics["mindts_semantic_mask_mean"] = float(np.mean(mindts_semantic_mask)) if mindts_semantic_mask else 0.0
    metrics["mindts_semantic_aux_loss"] = float(np.mean(mindts_semantic_aux_loss)) if mindts_semantic_aux_loss else 0.0
    metrics["retrieval_density_mean"] = float(np.mean(retrieval_density)) if retrieval_density else 0.0
    metrics["real_time_available_ratio"] = float(np.mean(real_time_available)) if real_time_available else 0.0
    metrics["retrieval_mean_age"] = float(np.mean(retrieval_mean_age)) if retrieval_mean_age else 0.0
    metrics["rasch_mastery_mean"] = float(np.mean(rasch_mastery)) if rasch_mastery else 0.0
    metrics["rasch_difficulty_mean"] = float(np.mean(rasch_difficulty)) if rasch_difficulty else 0.0
    metrics["rasch_discrimination_mean"] = (
        float(np.mean(rasch_discrimination)) if rasch_discrimination else 0.0
    )
    try:
        metrics["base_auc"] = float(roc_auc_score(all_labels, all_base_probs))
    except ValueError:
        metrics["base_auc"] = float("nan")
    try:
        metrics["semantic_auc"] = float(roc_auc_score(all_labels, all_semantic_probs))
    except ValueError:
        metrics["semantic_auc"] = float("nan")
    return metrics


def infer_prompt_dim(prompt_embedding_dir, fallback):
    if not prompt_embedding_dir:
        return fallback
    meta_path = Path(prompt_embedding_dir) / "train" / "metadata.json"
    root_meta_path = Path(prompt_embedding_dir) / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "embedding_dim" in meta:
            return int(meta["embedding_dim"])
    if root_meta_path.exists():
        meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
        if "embedding_dim" in meta:
            return int(meta["embedding_dim"])
    emb_path = Path(prompt_embedding_dir) / "train" / "embeddings.pt"
    if emb_path.exists():
        payload = torch.load(emb_path, map_location="cpu")
        return int(payload["embeddings"].shape[-1])
    pair_path = Path(prompt_embedding_dir) / "pair_embeddings.pt"
    if pair_path.exists():
        payload = torch.load(pair_path, map_location="cpu")
        return int(payload["pair_embeddings"].shape[-1])
    return fallback


def validate_modalities(args, datasets):
    graph_path = Path(args.graph_adjacency_path)
    if args.strict_trimodal and not graph_path.is_file():
        raise FileNotFoundError(f"Strict tri-modal mode requires graph adjacency: {graph_path}")
    if not args.strict_trimodal:
        return
    if not args.prompt_embedding_dir:
        raise ValueError("Strict tri-modal mode requires --prompt_embedding_dir")
    for split, dataset in datasets.items():
        if len(dataset) == 0 or "prompt_embeddings" not in dataset[0]:
            raise ValueError(f"Strict tri-modal mode has no semantic evidence for split={split}")


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for CPU execution")
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.prompt_dim = infer_prompt_dim(args.prompt_embedding_dir, args.prompt_dim)
    if args.graph_adjacency_path is None:
        suffix = f"_{args.processed_tag}" if args.processed_tag else ""
        args.graph_adjacency_path = str(
            Path(args.data_root) / "processed" / f"graph_adjacency_tensors{suffix}.pt"
        )

    metadata = load_assist2009_metadata(args.data_root, args.processed_tag)
    train_set = Assist2009KTDataset(
        data_root=args.data_root,
        split="train",
        processed_tag=args.processed_tag,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        test_ratio=args.test_ratio,
        num_folds=args.num_folds,
        fold_idx=args.fold_idx,
        max_samples=args.max_train_samples,
        sample_percent=args.sample_percent,
        prompt_embedding_dir=args.prompt_embedding_dir,
    )
    val_set = Assist2009KTDataset(
        data_root=args.data_root,
        split="val",
        processed_tag=args.processed_tag,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        test_ratio=args.test_ratio,
        num_folds=args.num_folds,
        fold_idx=args.fold_idx,
        max_samples=args.max_eval_samples,
        sample_percent=args.sample_percent,
        prompt_embedding_dir=args.prompt_embedding_dir,
    )
    test_set = Assist2009KTDataset(
        data_root=args.data_root,
        split="test",
        processed_tag=args.processed_tag,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        test_ratio=args.test_ratio,
        num_folds=args.num_folds,
        fold_idx=args.fold_idx,
        max_samples=args.max_eval_samples,
        sample_percent=args.sample_percent,
        prompt_embedding_dir=args.prompt_embedding_dir,
    )
    validate_modalities(args, {"train": train_set, "val": val_set, "test": test_set})
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    graph_path = args.graph_adjacency_path if Path(args.graph_adjacency_path).is_file() else None
    model = EvoTriKT(
        num_questions=metadata["question_vocab_size"],
        num_concepts=metadata["concept_vocab_size"],
        hidden_dim=args.hidden_dim,
        prompt_dim=args.prompt_dim,
        max_seq_len=args.max_seq_len,
        seq_layers=args.seq_layers,
        prompt_layers=args.prompt_layers,
        transformer_heads=args.transformer_heads,
        alignment_heads=args.alignment_heads,
        dropout=args.dropout,
        semantic_tokens=args.semantic_tokens,
        graph_adjacency_path=graph_path,
        use_state_forgetting=not args.disable_state_forgetting,
        semantic_fusion_mode=args.semantic_fusion_mode,
        semantic_modality_dropout=args.semantic_modality_dropout,
        question_dropout=args.question_dropout,
        folibi_alpha=args.folibi_alpha,
        target_residual_scale=args.target_residual_scale,
        use_response_interaction=args.use_response_interaction,
        response_interaction_scale=args.response_interaction_scale,
        response_interaction_mode=args.response_interaction_mode,
        response_interaction_dropout=args.response_interaction_dropout,
        use_rich_graph_evidence=args.use_rich_graph_evidence,
        use_target_denoised_pooling=args.use_target_denoised_pooling,
        use_student_semantic_adapter=args.use_student_semantic_adapter,
        student_semantic_rank=args.student_semantic_rank,
        student_semantic_dropout=args.student_semantic_dropout,
        student_semantic_max_scale=args.student_semantic_max_scale,
        student_semantic_gate_bias=args.student_semantic_gate_bias,
        student_semantic_adapter_mode=args.student_semantic_adapter_mode,
        use_semantic_advantage_calibration=args.use_semantic_advantage_calibration,
        semantic_advantage_max_delta=args.semantic_advantage_max_delta,
        use_mindts_semantic_readout=args.use_mindts_semantic_readout,
        use_view_residual_calibration=args.use_view_residual_calibration,
        view_residual_max_delta=args.view_residual_max_delta,
    ).to(device)
    if args.data_parallel:
        if device.type != "cuda":
            raise ValueError("--data_parallel requires a CUDA device")
        device_ids = parse_device_ids(args.device_ids)
        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) < 2:
            raise ValueError("--data_parallel requires at least two visible CUDA devices")
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print("data_parallel", True, "device_ids", device_ids, flush=True)
    else:
        print("data_parallel", False, flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ema_state = None
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    if args.warmup_steps > 0:
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_steps), eta_min=args.lr * 0.01)
        scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    print("device", device)
    print("metadata", metadata)
    print("samples", {"train": len(train_set), "val": len(val_set), "test": len(test_set)})
    print("prompt_embedding_dir", args.prompt_embedding_dir)
    print("prompt_dim", args.prompt_dim)
    print("strict_trimodal", args.strict_trimodal)
    print("graph_aux_weight", args.graph_aux_weight)
    print("prompt_aux_weight", args.prompt_aux_weight)
    print("view_consistency_teacher", args.view_consistency_teacher)
    print("view_consistency_weight", args.view_consistency_weight)
    print("validation_only", args.validation_only)
    print("ema_decay", args.ema_decay)
    print("ema_start_epoch", args.ema_start_epoch)
    print("semantic_fusion_mode", args.semantic_fusion_mode)
    print("semantic_modality_dropout", args.semantic_modality_dropout)
    print("question_dropout", args.question_dropout)
    print("folibi_alpha", args.folibi_alpha)
    print("target_residual_scale", args.target_residual_scale)
    print("use_response_interaction", args.use_response_interaction)
    print("response_interaction_scale", args.response_interaction_scale)
    print("response_interaction_mode", args.response_interaction_mode)
    print("response_interaction_dropout", args.response_interaction_dropout)
    print("use_rich_graph_evidence", args.use_rich_graph_evidence)
    print("use_target_denoised_pooling", args.use_target_denoised_pooling)
    print("use_student_semantic_adapter", args.use_student_semantic_adapter)
    print("student_semantic_rank", args.student_semantic_rank)
    print("student_semantic_dropout", args.student_semantic_dropout)
    print("student_semantic_max_scale", args.student_semantic_max_scale)
    print("student_semantic_gate_bias", args.student_semantic_gate_bias)
    print("student_semantic_gate_reg", args.student_semantic_gate_reg)
    print("student_semantic_delta_ratio_reg", args.student_semantic_delta_ratio_reg)
    print("student_semantic_adapter_mode", args.student_semantic_adapter_mode)
    print("use_semantic_advantage_calibration", args.use_semantic_advantage_calibration)
    print("semantic_advantage_max_delta", args.semantic_advantage_max_delta)
    print("semantic_advantage_lambda_reg", args.semantic_advantage_lambda_reg)
    print("use_view_residual_calibration", args.use_view_residual_calibration)
    print("view_residual_max_delta", args.view_residual_max_delta)
    print("view_residual_gate_reg", args.view_residual_gate_reg)
    print("view_residual_correction_reg", args.view_residual_correction_reg)
    print("use_mindts_semantic_readout", args.use_mindts_semantic_readout)
    print("mindts_semantic_aux_weight", args.mindts_semantic_aux_weight)
    print("graph_adjacency_path", args.graph_adjacency_path)
    print("params", sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad))

    best_auc = -1.0
    best_epoch = 0
    bad_epochs = 0
    best_path = Path(args.save_dir) / "best_model.pth"
    progress_path = Path(args.save_dir) / "metrics_so_far.json"
    history = []
    for epoch in range(1, args.epochs + 1):
        if args.ema_decay > 0.0 and ema_state is None and epoch >= args.ema_start_epoch:
            ema_state = init_ema(model)
        model.train()
        t1 = time.time()
        total_loss = 0.0
        total_count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            out = model(
                question_ids=batch["question_ids"],
                concept_ids=batch["concept_ids"],
                responses=batch["responses"],
                delta_t=batch["delta_t"],
                retention_age=batch.get("retention_age"),
                difficulty=batch["difficulty"],
                padding_mask=batch["padding_mask"],
                next_question_ids=batch["next_question_ids"],
                next_concept_ids=batch["next_concept_ids"],
                prompt_embeddings=get_prompt_embeddings(batch),
                history_prompt_embeddings=get_history_prompt_embeddings(batch),
                student_context_features=batch.get("student_context_features"),
                return_aux=True,
                **get_multi_concept_kwargs(model, batch),
            )
            main_loss = F.binary_cross_entropy_with_logits(out["logits"], batch["labels"])
            loss = main_loss
            if args.aux_loss_weight > 0:
                view_logits = [out["sequence_logits"], out["graph_logits"]]
                view_logits.append(out["prompt_logits"])
                aux_loss = sum(
                    F.binary_cross_entropy_with_logits(view, batch["labels"])
                    for view in view_logits
                ) / len(view_logits)
                loss = loss + args.aux_loss_weight * aux_loss
            if args.graph_aux_weight > 0:
                loss = loss + args.graph_aux_weight * F.binary_cross_entropy_with_logits(
                    out["graph_logits"], batch["labels"]
                )
            if args.prompt_aux_weight > 0 and "prompt_logits" in out:
                prompt_aux = F.binary_cross_entropy_with_logits(
                    out["prompt_logits"], batch["labels"], reduction="none"
                )
                if "semantic_available" in out:
                    sem_mask = out["semantic_available"].to(prompt_aux.dtype)
                    prompt_aux = (prompt_aux * sem_mask).sum() / sem_mask.sum().clamp_min(1.0)
                else:
                    prompt_aux = prompt_aux.mean()
                loss = loss + args.prompt_aux_weight * prompt_aux
            if args.view_consistency_weight > 0:
                teacher_prob = (
                    torch.sigmoid(out["logits"]).detach()
                    if args.view_consistency_teacher == "final"
                    else out["p_sequence"].detach()
                )
                graph_cons = F.binary_cross_entropy_with_logits(
                    out["graph_logits"], teacher_prob, reduction="none"
                ).mean()
                consistency_terms = [graph_cons]
                if "prompt_logits" in out:
                    prompt_cons = F.binary_cross_entropy_with_logits(
                        out["prompt_logits"], teacher_prob, reduction="none"
                    )
                    if "semantic_available" in out:
                        sem_mask = out["semantic_available"].to(prompt_cons.dtype)
                        prompt_cons = (prompt_cons * sem_mask).sum() / sem_mask.sum().clamp_min(1.0)
                    else:
                        prompt_cons = prompt_cons.mean()
                    consistency_terms.append(prompt_cons)
                loss = loss + args.view_consistency_weight * (
                    sum(consistency_terms) / len(consistency_terms)
                )
            if args.gate_balance_weight > 0:
                loss = loss + args.gate_balance_weight * (out["kar_gate"].mean() - 0.5).pow(2)
            if args.evidence_balance_weight > 0:
                loss = loss + args.evidence_balance_weight * (out["scaef_alpha"].mean() - 0.5).pow(2)
            if args.semantic_advantage_lambda_reg > 0 and "semantic_advantage_lambda" in out:
                loss = loss + args.semantic_advantage_lambda_reg * out["semantic_advantage_lambda"].mean()
            if args.student_semantic_gate_reg > 0 and "student_semantic_gate" in out:
                loss = loss + args.student_semantic_gate_reg * out["student_semantic_gate"].mean()
            if args.student_semantic_delta_ratio_reg > 0 and "student_semantic_delta_base_ratio" in out:
                loss = loss + args.student_semantic_delta_ratio_reg * out["student_semantic_delta_base_ratio"].mean()
            if args.mindts_semantic_aux_weight > 0 and "mindts_semantic_aux_loss" in out:
                loss = loss + args.mindts_semantic_aux_weight * out["mindts_semantic_aux_loss"].mean()
            if args.view_residual_gate_reg > 0 and "view_residual_gate" in out:
                loss = loss + args.view_residual_gate_reg * out["view_residual_gate"].mean()
            if args.view_residual_correction_reg > 0 and "view_residual_correction_abs" in out:
                loss = loss + args.view_residual_correction_reg * out["view_residual_correction_abs"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            if ema_state is not None:
                update_ema(model, ema_state, args.ema_decay)
            total_loss += float(loss.detach().cpu()) * batch["labels"].numel()
            total_count += batch["labels"].numel()

        val_metrics = evaluate_maybe_ema(model, val_loader, device, ema_state)
        train_loss = total_loss / max(total_count, 1)
        scheduler.step()
        elapsed = time.time() - t1
        row = {"epoch": epoch, "train_loss": train_loss, "val": val_metrics, "time": elapsed}
        history.append(row)
        print(
            f"Epoch {epoch:03d} train_loss={train_loss:.5f} "
            f"val_auc={val_metrics['auc']:.5f} val_acc={val_metrics['acc']:.5f} "
            f"val_rmse={val_metrics['rmse']:.5f} val_nll={val_metrics['nll']:.5f} "
            f"time={elapsed:.2f}s",
            flush=True,
        )
        improved = val_metrics["auc"] > best_auc + args.min_delta
        if improved:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            bad_epochs = 0
            if ema_state is None:
                save_model_state(model, best_path)
            else:
                backup = copy_ema_to_model(model, ema_state)
                save_model_state(model, best_path)
                restore_model_params(model, backup)
        else:
            bad_epochs += 1
        progress = {
            "args": vars(args),
            "metadata": metadata,
            "best_val_auc": best_auc,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "history": history,
        }
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        if args.patience > 0 and bad_epochs >= args.patience:
            print(
                f"Early stopping at epoch {epoch:03d}; "
                f"best_epoch={best_epoch:03d} best_val_auc={best_auc:.5f}",
                flush=True,
            )
            break

    load_model_state(model, best_path, device)
    test_metrics = None if args.validation_only else evaluate(model, test_loader, device)
    result = {"args": vars(args), "metadata": metadata, "best_val_auc": best_auc, "best_epoch": best_epoch, "test": test_metrics, "history": history}
    result_path = Path(args.save_dir) / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if test_metrics is None:
        print("Validation-only run: test split was not evaluated", flush=True)
    else:
        print("Test", json.dumps(test_metrics, indent=2), flush=True)
    print("saved", str(result_path), str(best_path), flush=True)


if __name__ == "__main__":
    main()
