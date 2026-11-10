import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def _dataset_name(root: Path) -> str:
    name = root.name.lower()
    if name.startswith("assist2009"):
        return "assist2009"
    if name == "eedi_neurips":
        return "nips34"
    if name not in {"assist2009", "nips34", "xes3g5m", "ednet"}:
        raise ValueError(f"Cannot infer paper-aligned dataset from {root}")
    return name


def load_paper_aligned_metadata(data_root: str, processed_tag: str = "paper_aligned_v1") -> Dict[str, int]:
    root = Path(data_root)
    dataset = _dataset_name(root)
    processed = root / "processed"
    maps = json.loads(
        (processed / f"{dataset}_paper_id_maps_{processed_tag}.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (processed / f"{dataset}_paper_summary_{processed_tag}.json").read_text(encoding="utf-8")
    )
    return {
        "question_vocab_size": int(maps["question_vocab_size"]),
        "concept_vocab_size": int(maps["concept_vocab_size"]),
        "student_count": int(summary["processed_users"]),
        "max_concepts": int(maps["max_concepts"]),
    }


class PaperAlignedKTDataset(Dataset):
    """Question-level pyKT folds with complete multi-KC inputs."""

    def __init__(
        self,
        data_root: str,
        split: str,
        max_seq_len: int = 200,
        seed: int = 1024,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.2,
        num_folds: int = 5,
        fold_idx: int = 0,
        processed_tag: str = "paper_aligned_v1",
        max_samples: Optional[int] = None,
        sample_percent: Optional[float] = None,
        prompt_embedding_dir: Optional[str] = None,
    ):
        del train_ratio, val_ratio, test_ratio, num_folds
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.root = Path(data_root)
        self.dataset_name = _dataset_name(self.root)
        self.split = split
        self.max_seq_len = int(max_seq_len)
        self.processed_tag = processed_tag
        processed = self.root / "processed"
        maps = json.loads(
            (processed / f"{self.dataset_name}_paper_id_maps_{processed_tag}.json").read_text(encoding="utf-8")
        )
        self.question_id_map = {str(k): int(v) for k, v in maps["question_id_map"].items()}
        self.concept_id_map = {str(k): int(v) for k, v in maps["concept_id_map"].items()}
        self.question_vocab_size = int(maps["question_vocab_size"])
        self.concept_vocab_size = int(maps["concept_vocab_size"])
        self.max_concepts = int(maps["max_concepts"])
        self.unk_question = self.question_id_map.get("[UNK]", 1)
        self.unk_concept = self.concept_id_map.get("[UNK]", 1)

        all_records = []
        sequence_path = processed / f"{self.dataset_name}_paper_sequences_{processed_tag}.jsonl"
        with sequence_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if int(record["sequence_length"]) >= 2:
                        all_records.append(record)

        self.sequences = []
        self.samples: List[Tuple[int, int]] = []
        self.split_user_ids = set()
        fold_idx = int(fold_idx) % 5
        for record in all_records:
            fixed_split = record.get("fixed_split")
            fold = int(record.get("fold", -1))
            include = (
                (split == "test" and fixed_split == "test")
                or (split == "val" and fixed_split == "trainval" and fold == fold_idx)
                or (split == "train" and fixed_split == "trainval" and fold != fold_idx)
            )
            if not include:
                continue
            seq_idx = len(self.sequences)
            self.sequences.append(record)
            self.split_user_ids.add(str(record["user_id"]))
            self.samples.extend((seq_idx, target) for target in range(1, int(record["sequence_length"])))

        rng = random.Random(seed + {"train": 0, "val": 1, "test": 2}[split])
        if sample_percent is not None and 0.0 < sample_percent < 1.0:
            keep = max(1, int(len(self.samples) * sample_percent))
            rng.shuffle(self.samples)
            self.samples = self.samples[:keep]
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = rng.sample(self.samples, max_samples)

        self.prompt_embeddings = None
        self.pair_prompt_embeddings = None
        self.pair_prompt_index = None
        if prompt_embedding_dir:
            prompt_root = Path(prompt_embedding_dir)
            pair_path = prompt_root / "pair_embeddings.pt"
            split_path = prompt_root / split / "embeddings.pt"
            if pair_path.exists():
                payload = torch.load(pair_path, map_location="cpu")
                self.pair_prompt_embeddings = payload["pair_embeddings"]
                self.pair_prompt_index = {
                    (int(q), int(c)): idx
                    for idx, (q, c) in enumerate(payload["pair_keys"].long().tolist())
                }
            elif split_path.exists():
                payload = torch.load(split_path, map_location="cpu")
                self.prompt_embeddings = payload["embeddings"]
                if self.prompt_embeddings.size(0) != len(self.samples):
                    raise ValueError(
                        f"Embedding count mismatch for {split}: {self.prompt_embeddings.size(0)} vs {len(self.samples)}"
                    )
            else:
                raise FileNotFoundError(f"No prompt embeddings found under {prompt_root}")

    def __len__(self):
        return len(self.samples)

    def _map_question(self, value):
        return self.question_id_map.get(str(value), self.unk_question)

    def _map_concept(self, value):
        return self.concept_id_map.get(str(value), self.unk_concept)

    @staticmethod
    def _gap_feature(seconds):
        seconds = max(float(seconds or 0.0), 0.0)
        return min(math.log1p(seconds) / math.log1p(30.0 * 24.0 * 3600.0), 1.0) if seconds else 0.0

    @staticmethod
    def _duration_feature(milliseconds):
        milliseconds = max(float(milliseconds or 0.0), 0.0)
        return min(math.log1p(milliseconds) / math.log1p(600000.0), 1.0) if milliseconds else 0.0

    def _mapped_set(self, record, position):
        values = record.get("concept_set_ids", [])[position]
        mapped = [self._map_concept(value) for value in values[: self.max_concepts]]
        return mapped

    def _student_context(self, record, positions, target_pos):
        if not positions:
            return torch.zeros(8, dtype=torch.float32)
        responses = [float(record["responses"][pos]) for pos in positions]
        recent = responses[-20:]
        target_q = str(record["question_ids"][target_pos])
        target_concepts = set(map(str, record["concept_set_ids"][target_pos]))
        related = [
            float(record["responses"][pos])
            for pos in positions
            if target_concepts.intersection(map(str, record["concept_set_ids"][pos]))
        ]
        question_seen = sum(str(record["question_ids"][pos]) == target_q for pos in positions)
        difficulty = [self._duration_feature(record["duration_ms"][pos]) for pos in positions]
        return torch.tensor(
            [
                min(len(positions) / self.max_seq_len, 1.0),
                sum(responses) / len(responses),
                sum(recent) / len(recent),
                sum(related) / len(related) if related else 0.5,
                min(math.log1p(len(related)) / math.log1p(self.max_seq_len), 1.0),
                min(math.log1p(question_seen) / math.log1p(20.0), 1.0),
                self._gap_feature(record["time_gap_seconds"][positions[-1]]),
                sum(difficulty) / len(difficulty),
            ],
            dtype=torch.float32,
        )

    def __getitem__(self, index):
        seq_idx, target_pos = self.samples[index]
        record = self.sequences[seq_idx]
        start = max(0, target_pos - self.max_seq_len)
        positions = list(range(start, target_pos))
        hist_len = len(positions)

        question_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
        concept_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
        concept_set_ids = torch.zeros(self.max_seq_len, self.max_concepts, dtype=torch.long)
        concept_set_mask = torch.zeros(self.max_seq_len, self.max_concepts, dtype=torch.bool)
        responses = torch.zeros(self.max_seq_len, dtype=torch.long)
        delta_t = torch.zeros(self.max_seq_len, dtype=torch.float32)
        retention_age = torch.zeros(self.max_seq_len, dtype=torch.float32)
        difficulty = torch.zeros(self.max_seq_len, dtype=torch.float32)
        padding_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)

        for out_pos, pos in enumerate(positions):
            question_ids[out_pos] = self._map_question(record["question_ids"][pos])
            mapped_set = self._mapped_set(record, pos)
            concept_ids[out_pos] = mapped_set[0] if mapped_set else self.unk_concept
            if mapped_set:
                concept_set_ids[out_pos, : len(mapped_set)] = torch.tensor(mapped_set)
                concept_set_mask[out_pos, : len(mapped_set)] = True
            responses[out_pos] = int(record["responses"][pos])
            delta_t[out_pos] = self._gap_feature(record["time_gap_seconds"][pos])
            difficulty[out_pos] = self._duration_feature(record["duration_ms"][pos])
            padding_mask[out_pos] = True

        if hist_len:
            raw_gaps = torch.tensor(record["time_gap_seconds"][start + 1 : target_pos + 1], dtype=torch.float64)
            ages = torch.flip(torch.cumsum(torch.flip(raw_gaps.clamp_min(0), dims=[0]), dim=0), dims=[0])
            retention_age[:hist_len] = torch.clamp(
                torch.log1p(ages / 86400.0) / math.log1p(365.0), max=1.0
            ).float()

        target_set = self._mapped_set(record, target_pos)
        next_concept_set_ids = torch.zeros(self.max_concepts, dtype=torch.long)
        next_concept_set_mask = torch.zeros(self.max_concepts, dtype=torch.bool)
        if target_set:
            next_concept_set_ids[: len(target_set)] = torch.tensor(target_set)
            next_concept_set_mask[: len(target_set)] = True
        next_q = self._map_question(record["question_ids"][target_pos])
        next_c = target_set[0] if target_set else self.unk_concept
        sample = {
            "sample_index": torch.tensor(index),
            "question_ids": question_ids,
            "concept_ids": concept_ids,
            "concept_set_ids": concept_set_ids,
            "concept_set_mask": concept_set_mask,
            "responses": responses,
            "delta_t": delta_t,
            "retention_age": retention_age,
            "difficulty": difficulty,
            "padding_mask": padding_mask,
            "next_question_ids": torch.tensor(next_q),
            "next_concept_ids": torch.tensor(next_c),
            "next_concept_set_ids": next_concept_set_ids,
            "next_concept_set_mask": next_concept_set_mask,
            "labels": torch.tensor(float(record["responses"][target_pos]), dtype=torch.float32),
            "history_len": torch.tensor(hist_len),
            "user_id": torch.tensor(int(record["user_id"])),
            "student_context_features": self._student_context(record, positions, target_pos),
        }
        if self.prompt_embeddings is not None:
            sample["prompt_embeddings"] = self.prompt_embeddings[index]
        elif self.pair_prompt_embeddings is not None:
            pair_index = self.pair_prompt_index.get((next_q, next_c))
            if pair_index is None:
                raise KeyError(f"Missing semantic embedding for mapped pair {(next_q, next_c)}")
            sample["prompt_embeddings"] = self.pair_prompt_embeddings[pair_index]
            dim = int(self.pair_prompt_embeddings.size(-1))
            history = torch.zeros(self.max_seq_len, dim, dtype=self.pair_prompt_embeddings.dtype)
            for out_pos, pos in enumerate(positions):
                hist_set = self._mapped_set(record, pos)
                hist_c = hist_set[0] if hist_set else self.unk_concept
                hist_q = self._map_question(record["question_ids"][pos])
                hist_idx = self.pair_prompt_index.get((hist_q, hist_c))
                if hist_idx is not None:
                    history[out_pos] = self.pair_prompt_embeddings[hist_idx]
            sample["history_prompt_embeddings"] = history
        return sample
