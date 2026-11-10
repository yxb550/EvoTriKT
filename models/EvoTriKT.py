import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _last_valid_index(mask: torch.Tensor) -> torch.Tensor:
    lengths = mask.long().sum(dim=1).clamp_min(1)
    return lengths - 1


def _gather_last(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    batch = sequence.size(0)
    idx = _last_valid_index(mask)
    return sequence[torch.arange(batch, device=sequence.device), idx]


def _binary_js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    p_dist = torch.stack([p, 1.0 - p], dim=-1)
    q_dist = torch.stack([q, 1.0 - q], dim=-1)
    m_dist = 0.5 * (p_dist + q_dist)
    kl_pm = (p_dist * (p_dist.log() - m_dist.log())).sum(dim=-1)
    kl_qm = (q_dist * (q_dist.log() - m_dist.log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


class KANLinear(nn.Module):
    """A lightweight KAN-style layer using fixed radial basis expansions."""

    def __init__(self, in_features: int, out_features: int, grid_size: int = 5):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.spline = nn.Linear(in_features * grid_size, out_features)
        centers = torch.linspace(-1.0, 1.0, grid_size)
        self.register_buffer("centers", centers)
        self.scale = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = torch.tanh(x)
        basis = torch.exp(-((x_norm.unsqueeze(-1) - self.centers) ** 2) / (2.0 * self.scale.abs().clamp_min(1e-3) ** 2))
        return self.base(x) + self.spline(basis.flatten(start_dim=-2))


class KnowledgeInteractionSequenceEncoder(nn.Module):
    """Causal encoder for raw knowledge tracing interaction histories."""

    def __init__(
        self,
        num_questions: int,
        num_concepts: int,
        hidden_dim: int,
        max_seq_len: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        question_dropout: float = 0.0,
        folibi_alpha: float = 0.0,
        use_response_interaction: bool = False,
        response_interaction_scale: float = 0.25,
        response_interaction_mode: str = "concept",
        response_interaction_dropout: float = 0.0,
    ):
        super().__init__()
        self.question_dropout = float(question_dropout)
        self.folibi_alpha = float(folibi_alpha)
        self.use_response_interaction = bool(use_response_interaction)
        if response_interaction_mode not in {"concept", "question_concept"}:
            raise ValueError("response_interaction_mode must be 'concept' or 'question_concept'")
        self.response_interaction_mode = response_interaction_mode
        self.response_interaction_scale = float(response_interaction_scale)
        self.response_interaction_dropout = nn.Dropout(float(response_interaction_dropout))
        self.max_seq_len = int(max_seq_len)
        self.question_emb = nn.Embedding(num_questions, hidden_dim, padding_idx=0)
        self.concept_emb = nn.Embedding(num_concepts, hidden_dim, padding_idx=0)
        self.response_emb = nn.Embedding(3, hidden_dim, padding_idx=2)
        self.question_response_emb = (
            nn.Embedding(num_questions * 2 + 2, hidden_dim, padding_idx=0)
            if self.use_response_interaction and response_interaction_mode == "question_concept"
            else None
        )
        self.concept_response_emb = (
            nn.Embedding(num_concepts * 2 + 2, hidden_dim, padding_idx=0)
            if self.use_response_interaction
            else None
        )
        self.position_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.delta_proj = nn.Linear(1, hidden_dim)
        self.difficulty_proj = nn.Linear(1, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        delta_t: Optional[torch.Tensor],
        difficulty: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        return_sequence: bool = False,
        concept_set_ids: Optional[torch.Tensor] = None,
        concept_set_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len = question_ids.shape
        device = question_ids.device
        if padding_mask is None:
            padding_mask = question_ids.ne(0)
        if delta_t is None:
            delta_t = torch.zeros(batch, seq_len, device=device)
        if difficulty is None:
            difficulty = torch.zeros(batch, seq_len, device=device)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        response_ids = responses.long().clamp(0, 1).masked_fill(~padding_mask, 2)
        question_emb = self.question_emb(question_ids.clamp_min(0))
        if self.training and self.question_dropout > 0.0:
            keep = 1.0 - self.question_dropout
            drop_mask = torch.empty(
                batch, seq_len, 1, device=device, dtype=question_emb.dtype
            ).bernoulli_(keep)
            question_emb = question_emb * drop_mask / max(keep, 1e-6)
        concept_emb = self.concept_emb(concept_ids.clamp_min(0))
        if concept_set_ids is not None:
            set_ids = concept_set_ids.clamp(0, self.concept_emb.num_embeddings - 1)
            set_emb = self.concept_emb(set_ids)
            if concept_set_mask is None:
                concept_set_mask = concept_set_ids.gt(0)
            weights = concept_set_mask.to(set_emb.dtype).unsqueeze(-1)
            concept_emb = (set_emb * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
        response_emb = self.response_emb(response_ids)
        if self.use_response_interaction and self.concept_response_emb is not None:
            valid_response = response_ids.clamp(0, 1)
            cr_ids = torch.where(
                padding_mask,
                concept_ids.clamp(0, self.concept_emb.num_embeddings - 1) * 2 + valid_response + 1,
                torch.zeros_like(concept_ids),
            )
            interaction_emb = self.concept_response_emb(cr_ids)
            if self.question_response_emb is not None:
                qr_ids = torch.where(
                    padding_mask,
                    question_ids.clamp(0, self.question_emb.num_embeddings - 1) * 2 + valid_response + 1,
                    torch.zeros_like(question_ids),
                )
                interaction_emb = interaction_emb + self.question_response_emb(qr_ids)
            interaction_emb = self.response_interaction_dropout(interaction_emb)
            response_emb = response_emb + self.response_interaction_scale * interaction_emb
        x = (
            question_emb
            + concept_emb
            + response_emb
            + self.position_emb(positions)
            + self.delta_proj(delta_t.float().unsqueeze(-1))
            + self.difficulty_proj(difficulty.float().unsqueeze(-1))
        )
        if self.folibi_alpha > 0.0:
            query_pos = torch.arange(seq_len, device=device).unsqueeze(1)
            key_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            distance = (query_pos - key_pos).clamp_min(0).float()
            distance = distance / max(float(self.max_seq_len - 1), 1.0)
            causal_mask = -self.folibi_alpha * distance
            causal_mask = causal_mask.masked_fill(key_pos > query_pos, float("-inf"))
        else:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1,
            )
        encoded = self.encoder(x, mask=causal_mask, src_key_padding_mask=~padding_mask)
        encoded = self.norm(encoded)
        if return_sequence:
            return encoded
        return _gather_last(encoded, padding_mask)


class CausalStudentKnowledgeStateUpdater(nn.Module):
    """Writes evidence into concept states with explicit continuous forgetting."""

    def __init__(
        self,
        num_atomic_concepts: int,
        hidden_dim: int,
        question_to_atomic_indices: torch.Tensor,
        concept_to_atomic_indices: torch.Tensor,
        cooc_norm: torch.Tensor,
        use_state_forgetting: bool = True,
        use_rich_graph_evidence: bool = False,
    ):
        super().__init__()
        self.num_atomic_concepts = num_atomic_concepts
        self.use_state_forgetting = use_state_forgetting
        self.use_rich_graph_evidence = bool(use_rich_graph_evidence)
        self.register_buffer("question_to_atomic", question_to_atomic_indices.long())
        self.register_buffer("concept_to_atomic", concept_to_atomic_indices.long())
        self.register_buffer("cooc_norm", cooc_norm.float())
        self.atomic_concept_emb = nn.Embedding(num_atomic_concepts, hidden_dim)
        # softplus(0) ~= log(2), so an age of 1 starts at 50% retention.
        # Each atomic concept learns its own non-negative forgetting rate.
        self.forget_log_rate = nn.Parameter(torch.zeros(num_atomic_concepts))
        self.evidence_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.rich_evidence_proj = (
            nn.Sequential(
                nn.Linear(hidden_dim + 4, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            if self.use_rich_graph_evidence
            else None
        )
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.propagate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        rate_key = prefix + "forget_log_rate"
        if rate_key not in state_dict:
            state_dict[rate_key] = self.forget_log_rate.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _decay_toward_baseline(
        self,
        state: torch.Tensor,
        baseline: torch.Tensor,
        atomic_ids: torch.Tensor,
        elapsed_age: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rates = F.softplus(self.forget_log_rate)[atomic_ids]
        retention = torch.exp(-rates * elapsed_age.clamp_min(0.0))
        decayed = baseline + retention.unsqueeze(-1) * (state - baseline)
        return decayed, retention

    @staticmethod
    def _infer_target_relative_age(
        delta_t: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Fallback for callers that only provide adjacent event gaps."""
        valid_delta = delta_t.float().clamp_min(0.0) * padding_mask.float()
        reverse_cumulative = torch.flip(
            torch.cumsum(torch.flip(valid_delta, dims=[1]), dim=1), dims=[1]
        )
        return (reverse_cumulative - valid_delta).clamp_min(0.0) * padding_mask.float()

    def _atomic_ids(self, question_ids: torch.Tensor, concept_ids: torch.Tensor) -> torch.Tensor:
        q_ids = question_ids.clamp(0, self.question_to_atomic.size(0) - 1)
        c_ids = concept_ids.clamp(0, self.concept_to_atomic.size(0) - 1)
        q_atomics = F.embedding(q_ids, self.question_to_atomic)
        c_atomics = F.embedding(c_ids, self.concept_to_atomic)
        width = max(q_atomics.size(-1), c_atomics.size(-1))
        if q_atomics.size(-1) < width:
            q_atomics = F.pad(q_atomics, (0, width - q_atomics.size(-1)), value=-1)
        if c_atomics.size(-1) < width:
            c_atomics = F.pad(c_atomics, (0, width - c_atomics.size(-1)), value=-1)
        q_has_mapping = q_atomics.ge(0).any(dim=-1, keepdim=True)
        return torch.where(q_has_mapping, q_atomics, c_atomics)

    def forward(
        self,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        delta_t: Optional[torch.Tensor],
        difficulty: Optional[torch.Tensor],
        padding_mask: torch.Tensor,
        retention_age: Optional[torch.Tensor] = None,
        event_evidence_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch, seq_len = concept_ids.shape
        device = concept_ids.device
        hidden_dim = self.atomic_concept_emb.embedding_dim
        if delta_t is None:
            delta_t = torch.zeros(batch, seq_len, device=device)
        if retention_age is None:
            retention_age = self._infer_target_relative_age(delta_t, padding_mask)
        else:
            retention_age = retention_age.to(device=device, dtype=torch.float32)
        if difficulty is None:
            difficulty = torch.zeros(batch, seq_len, device=device)

        atomics = self._atomic_ids(question_ids, concept_ids)
        baseline = self.atomic_concept_emb.weight.unsqueeze(0).expand(batch, -1, -1)
        node_state = baseline.clone()
        visit_count = torch.zeros(batch, self.num_atomic_concepts, device=device)
        last_update_age = torch.zeros(batch, self.num_atomic_concepts, device=device)
        retention_sum = torch.zeros((), device=device)
        retention_count = torch.zeros((), device=device)

        # Sequential writes preserve a causal student state rather than a bag of attempts.
        for step in range(seq_len):
            step_atomics = atomics[:, step]
            valid = step_atomics.ge(0) & padding_mask[:, step].unsqueeze(-1)
            safe_atomics = step_atomics.masked_fill(~valid, 0)
            response = responses[:, step].float().clamp(0.0, 1.0)
            recency = (
                torch.ones_like(response)
                if self.use_state_forgetting
                else torch.exp(-delta_t[:, step].float().clamp_min(0.0))
            )
            features = torch.stack(
                [response, 1.0 - response, recency, difficulty[:, step].float()], dim=-1
            )
            if (
                self.use_rich_graph_evidence
                and self.rich_evidence_proj is not None
                and event_evidence_state is not None
            ):
                evidence = self.rich_evidence_proj(
                    torch.cat([event_evidence_state[:, step], features], dim=-1)
                )
            else:
                evidence = self.evidence_proj(features)
            atom_count = valid.sum(dim=1, keepdim=True).clamp_min(1).to(evidence.dtype)
            evidence = evidence / atom_count

            previous_raw = torch.gather(
                node_state,
                1,
                safe_atomics.unsqueeze(-1).expand(-1, -1, hidden_dim),
            )
            previous = previous_raw
            previous_baseline = torch.gather(
                baseline,
                1,
                safe_atomics.unsqueeze(-1).expand(-1, -1, hidden_dim),
            )
            previous_age = torch.gather(last_update_age, 1, safe_atomics)
            current_age = retention_age[:, step].unsqueeze(1).expand_as(previous_age)
            was_visited = torch.gather(visit_count, 1, safe_atomics).gt(0)
            elapsed_age = torch.where(
                was_visited,
                (previous_age - current_age).clamp_min(0.0),
                torch.zeros_like(current_age),
            )
            if self.use_state_forgetting:
                previous, retention = self._decay_toward_baseline(
                    previous, previous_baseline, safe_atomics, elapsed_age
                )
                retention_sum = retention_sum + (retention * valid).sum()
                retention_count = retention_count + valid.sum()
            updated = self.node_update(
                evidence.unsqueeze(1).expand_as(previous).reshape(-1, hidden_dim),
                previous.reshape(-1, hidden_dim),
            ).view_as(previous)
            delta = (updated - previous_raw) * valid.unsqueeze(-1)
            state_delta = torch.zeros_like(node_state).scatter_add(
                1,
                safe_atomics.unsqueeze(-1).expand(-1, -1, hidden_dim),
                delta,
            )
            node_state = node_state + state_delta
            visit_count.scatter_add_(1, safe_atomics, valid.float())
            age_delta = (current_age - previous_age) * valid
            last_update_age = last_update_age + torch.zeros_like(last_update_age).scatter_add(
                1, safe_atomics, age_delta
            )

        if self.use_state_forgetting:
            all_atomic_ids = torch.arange(
                self.num_atomic_concepts, device=device
            ).unsqueeze(0).expand(batch, -1)
            node_state, target_retention = self._decay_toward_baseline(
                node_state, baseline, all_atomic_ids, last_update_age
            )
            visited = visit_count.gt(0)
            retention_sum = retention_sum + (target_retention * visited).sum()
            retention_count = retention_count + visited.sum()

        relation = self.cooc_norm.to(device=device, dtype=node_state.dtype)
        neighbor_state = torch.matmul(relation, node_state)
        node_state = node_state + self.propagate(torch.cat([node_state, neighbor_state], dim=-1))
        graph_density = visit_count.gt(0).float().mean(dim=1)
        history_len = padding_mask.float().sum(dim=1)
        confidence = torch.sigmoid(torch.log1p(history_len) / 4.0 + graph_density)
        mean_retention = retention_sum / retention_count.clamp_min(1.0)
        return {
            "node_state": node_state,
            "confidence": confidence,
            "history_len": history_len,
            "visit_count": visit_count,
            "mean_retention": mean_retention,
        }


class StudentKnowledgeStateGraphReadout(nn.Module):
    """Reads global and target-local evidence from a student knowledge-state graph."""

    def __init__(
        self,
        hidden_dim: int,
        num_atomic_concepts: int,
        question_to_atomic_indices: torch.Tensor,
        concept_to_atomic_indices: torch.Tensor,
    ):
        super().__init__()
        self.num_atomic_concepts = num_atomic_concepts
        # The adjacency is reconstructed at startup, so keeping this buffer
        # non-persistent preserves compatibility with earlier checkpoints.
        self.register_buffer("question_to_atomic", question_to_atomic_indices.long(), persistent=False)
        self.register_buffer("concept_to_atomic", concept_to_atomic_indices.long())
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(
        self,
        node_state: torch.Tensor,
        next_question_ids: torch.Tensor,
        next_concept_ids: torch.Tensor,
        next_concept_emb: torch.Tensor,
    ) -> torch.Tensor:
        q = self.query_proj(next_concept_emb).unsqueeze(1)
        k = self.key_proj(node_state)
        v = self.value_proj(node_state)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1)), dim=-1)
        graph_context = torch.matmul(attn, v).squeeze(1)

        question_ids = next_question_ids.clamp(0, self.question_to_atomic.size(0) - 1)
        concept_ids = next_concept_ids.clamp(0, self.concept_to_atomic.size(0) - 1)
        question_atomics = F.embedding(question_ids, self.question_to_atomic)
        concept_atomics = F.embedding(concept_ids, self.concept_to_atomic)
        if concept_atomics.size(-1) < question_atomics.size(-1):
            concept_atomics = F.pad(
                concept_atomics, (0, question_atomics.size(-1) - concept_atomics.size(-1)), value=-1
            )
        question_has_mapping = question_atomics.ge(0).any(dim=-1, keepdim=True)
        atomics = torch.where(question_has_mapping, question_atomics, concept_atomics)
        valid = atomics.ge(0)
        safe_atomics = atomics.masked_fill(~valid, 0)
        target_nodes = torch.gather(
            node_state,
            1,
            safe_atomics.unsqueeze(-1).expand(-1, -1, node_state.size(-1)),
        )
        target_state = (target_nodes * valid.unsqueeze(-1)).sum(dim=1)
        target_state = target_state / valid.sum(dim=1, keepdim=True).clamp_min(1).to(target_state.dtype)
        return self.out(torch.cat([graph_context, target_state, next_concept_emb], dim=-1))


# Backward-compatible imports for earlier experiment scripts and checkpoints.
OnlineSequenceToGraphTransformer = CausalStudentKnowledgeStateUpdater
DynamicKnowledgeGraphEncoder = StudentKnowledgeStateGraphReadout


class SemanticPromptEncoder(nn.Module):
    """Turns frozen LLM evidence into multiple trainable semantic evidence tokens."""

    def __init__(
        self,
        prompt_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        semantic_tokens: int = 8,
    ):
        super().__init__()
        self.semantic_tokens = semantic_tokens
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)
        self.pooled_to_tokens = nn.Sequential(
            nn.LayerNorm(prompt_dim),
            nn.Linear(prompt_dim, hidden_dim * semantic_tokens),
        )
        self.token_type = nn.Parameter(torch.randn(1, semantic_tokens, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.missing_token = nn.Parameter(torch.zeros(1, semantic_tokens, hidden_dim))

    def forward(self, prompt_embeddings: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if prompt_embeddings is None:
            return self.missing_token.expand(batch_size, -1, -1)
        prompt_embeddings = prompt_embeddings.to(device=device, dtype=torch.float32, non_blocking=True)
        if prompt_embeddings.dim() == 2:
            tokens = self.pooled_to_tokens(prompt_embeddings).view(batch_size, self.semantic_tokens, -1)
            return self.encoder(tokens + self.token_type)
        return self.encoder(self.prompt_proj(prompt_embeddings))


class StudentStateAwareTemporalStructuralGate(nn.Module):
    """Replaces horizon-aware gating with student-state-aware sequence-graph routing."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        seq_state: torch.Tensor,
        graph_state: torch.Tensor,
        next_question_emb: torch.Tensor,
        next_concept_emb: torch.Tensor,
        last_delta_t: torch.Tensor,
        graph_confidence: torch.Tensor,
        history_len: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scalar_state = torch.stack(
            [
                torch.log1p(last_delta_t.float().clamp_min(0.0)),
                graph_confidence.float(),
                torch.log1p(history_len.float()),
            ],
            dim=-1,
        )
        gate = self.gate(torch.cat([seq_state, graph_state, next_question_emb, next_concept_emb, scalar_state], dim=-1))
        fused = gate * graph_state + (1.0 - gate) * seq_state
        return {"state": fused, "gate": gate}


class CausalMultiHeadCrossModalAlignment(nn.Module):
    """Aligns current student state with semantic prompt evidence without future responses."""

    def __init__(self, hidden_dim: int, num_alignment_heads: int = 4, attn_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.MultiheadAttention(hidden_dim, attn_heads, dropout=dropout, batch_first=True) for _ in range(num_alignment_heads)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = state.unsqueeze(1)
        outputs = []
        key_padding_mask = None if prompt_mask is None else ~prompt_mask
        for attn in self.heads:
            aligned, _ = attn(
                query,
                prompt_tokens,
                prompt_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            outputs.append(self.norm(aligned.squeeze(1) + state))
        return torch.stack(outputs, dim=1)


class KnowledgeAdaptiveHeadRouter(nn.Module):
    """KAN-style router for instance-level alignment-head importance."""

    def __init__(self, hidden_dim: int, num_alignment_heads: int = 4):
        super().__init__()
        self.num_alignment_heads = num_alignment_heads
        self.router = nn.Sequential(
            KANLinear(hidden_dim * num_alignment_heads + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_alignment_heads),
        )

    def forward(self, head_outputs: torch.Tensor, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        flat_heads = head_outputs.flatten(start_dim=1)
        weights = torch.softmax(self.router(torch.cat([flat_heads, context], dim=-1)), dim=-1)
        fused = torch.sum(weights.unsqueeze(-1) * head_outputs, dim=1)
        return {"state": fused, "weights": weights}


class EvidencePreservingDynamicResidualFusion(nn.Module):
    """Keeps behavioral-structural evidence when semantic alignment is noisy or conflicting."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gamma = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        base_state: torch.Tensor,
        aligned_state: torch.Tensor,
        next_question_emb: torch.Tensor,
        conflict_score: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gamma = self.gamma(
            torch.cat(
                [base_state, aligned_state, next_question_emb, torch.abs(base_state - aligned_state), conflict_score.unsqueeze(-1)],
                dim=-1,
            )
        )
        fused = gamma * aligned_state + (1.0 - gamma) * base_state
        return {"state": fused, "gamma": gamma}


class StudentStateGatedSemanticFusion(nn.Module):
    """Adds semantic evidence only when it agrees with the current student state."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        modality_dropout: float = 0.2,
    ):
        super().__init__()
        self.modality_dropout = float(modality_dropout)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.semantic_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.reliability = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        # Start near alpha=0.12 so the behavioral state remains the default.
        nn.init.constant_(self.reliability[-1].bias, -2.0)
        self.semantic_dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        base_state: torch.Tensor,
        prompt_tokens: torch.Tensor,
        next_question_emb: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None,
        semantic_available: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        key_padding_mask = None if prompt_mask is None else ~prompt_mask
        aligned, token_weights = self.attention(
            self.query_norm(base_state).unsqueeze(1),
            self.semantic_norm(prompt_tokens),
            self.semantic_norm(prompt_tokens),
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        aligned = aligned.squeeze(1)
        alpha = torch.sigmoid(
            self.reliability(
                torch.cat(
                    [base_state, aligned, torch.abs(base_state - aligned), next_question_emb],
                    dim=-1,
                )
            )
        )
        active = torch.ones(
            alpha.size(0), 1, device=alpha.device, dtype=torch.bool
        )
        if semantic_available is not None:
            active = active & semantic_available.unsqueeze(-1)
            alpha = alpha * semantic_available.to(alpha.dtype).unsqueeze(-1)
        if self.training and self.modality_dropout > 0.0:
            keep = torch.rand(alpha.size(0), 1, device=alpha.device).ge(
                self.modality_dropout
            )
            active = active & keep
            alpha = alpha * keep.to(alpha.dtype)
        candidate = self.out_norm(
            base_state + alpha * self.semantic_dropout(aligned)
        )
        fused = torch.where(active, candidate, base_state)
        return {
            "state": fused,
            "gamma": alpha,
            "weights": token_weights.squeeze(1),
            "aligned_state": aligned,
        }


class KnowledgeStatePredictionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, next_question_emb: torch.Tensor, next_concept_emb: torch.Tensor) -> torch.Tensor:
        return self.pred(torch.cat([state, next_question_emb, next_concept_emb], dim=-1)).squeeze(-1)


class SemanticAdvantageCalibration(nn.Module):
    """Routes bounded semantic logit corrections on top of the behavioral base."""

    def __init__(self, hidden_dim: int, max_delta: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.max_delta = float(max_delta)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.router[-2].bias, -1.0)

    def forward(
        self,
        base_state: torch.Tensor,
        semantic_state: torch.Tensor,
        seq_state: torch.Tensor,
        graph_state: torch.Tensor,
        prompt_state: torch.Tensor,
        base_logit: torch.Tensor,
        semantic_logit: torch.Tensor,
        p_sequence: torch.Tensor,
        p_graph: torch.Tensor,
        p_prompt: torch.Tensor,
        conflict: torch.Tensor,
        graph_confidence: torch.Tensor,
        semantic_available: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        features = torch.stack(
            [
                conflict,
                graph_confidence,
                torch.abs(p_prompt - p_sequence),
                torch.abs(p_prompt - p_graph),
            ],
            dim=-1,
        )
        route_input = torch.cat(
            [
                base_state,
                semantic_state,
                seq_state,
                graph_state,
                prompt_state,
                features,
            ],
            dim=-1,
        )
        lam = self.router(route_input).squeeze(-1)
        lam = lam * semantic_available.to(device=lam.device, dtype=lam.dtype)
        raw_correction = semantic_logit - base_logit
        bounded_correction = self.max_delta * torch.tanh(raw_correction)
        logits = base_logit + lam * bounded_correction
        return {
            "logits": logits,
            "lambda": lam,
            "raw_correction": raw_correction,
            "bounded_correction": bounded_correction,
        }


class TriViewResidualLogitCalibration(nn.Module):
    """Learns small expert-logit corrections without overwriting the strong base."""

    def __init__(self, hidden_dim: int, max_delta: float = 0.6, dropout: float = 0.1):
        super().__init__()
        self.max_delta = float(max_delta)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 7, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.constant_(self.router[-1].bias, -3.0)

    def forward(
        self,
        base_state: torch.Tensor,
        seq_state: torch.Tensor,
        graph_state: torch.Tensor,
        prompt_state: torch.Tensor,
        next_question_emb: torch.Tensor,
        base_logit: torch.Tensor,
        sequence_logit: torch.Tensor,
        graph_logit: torch.Tensor,
        prompt_logit: torch.Tensor,
        conflict: torch.Tensor,
        graph_confidence: torch.Tensor,
        semantic_available: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        p_base = torch.sigmoid(base_logit)
        p_seq = torch.sigmoid(sequence_logit)
        p_graph = torch.sigmoid(graph_logit)
        p_prompt = torch.sigmoid(prompt_logit)
        features = torch.stack(
            [
                conflict,
                graph_confidence,
                semantic_available.to(base_logit.dtype),
                torch.abs(p_seq - p_base),
                torch.abs(p_graph - p_base),
                torch.abs(p_prompt - p_base),
                torch.abs(p_graph - p_seq),
            ],
            dim=-1,
        )
        route_input = torch.cat(
            [base_state, seq_state, graph_state, prompt_state, next_question_emb, features],
            dim=-1,
        )
        gates = torch.sigmoid(self.router(route_input))
        gates = gates.clone()
        gates[:, 1] = gates[:, 1] * graph_confidence.clamp(0.0, 1.0)
        gates[:, 2] = gates[:, 2] * semantic_available.to(gates.dtype)
        expert_logits = torch.stack([sequence_logit, graph_logit, prompt_logit], dim=-1)
        bounded_delta = self.max_delta * torch.tanh(expert_logits - base_logit.unsqueeze(-1))
        correction = (gates * bounded_delta).sum(dim=-1) / gates.sum(dim=-1).clamp_min(1.0)
        logits = base_logit + correction
        return {
            "logits": logits,
            "gates": gates,
            "correction": correction,
            "bounded_delta": bounded_delta,
        }


class ConceptAnchoredTargetAdapter(nn.Module):
    """Keeps target-item information as a bounded residual around concept evidence."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1, residual_scale: float = 0.5):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        question_emb: torch.Tensor,
        concept_emb: torch.Tensor,
        question_dropout: float = 0.0,
    ) -> torch.Tensor:
        if self.training and question_dropout > 0.0:
            keep = 1.0 - float(question_dropout)
            drop_mask = torch.empty(
                question_emb.size(0), 1, device=question_emb.device, dtype=question_emb.dtype
            ).bernoulli_(keep)
            question_emb = question_emb * drop_mask / max(keep, 1e-6)
        joint = torch.cat([question_emb, concept_emb], dim=-1)
        return concept_emb + self.residual_scale * self.gate(joint) * self.residual(joint)


class TargetAwareDenoisedHistoryPooling(nn.Module):
    """Target-aware history attention with relevance and reliability biases."""

    def __init__(self, hidden_dim: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.reliability = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.same_question_bias = nn.Parameter(torch.tensor(0.2))
        self.same_concept_bias = nn.Parameter(torch.tensor(0.5))
        self.recency_scale = nn.Parameter(torch.tensor(0.1))
        self.mix_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        encoded_history: torch.Tensor,
        base_state: torch.Tensor,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        delta_t: Optional[torch.Tensor],
        difficulty: Optional[torch.Tensor],
        padding_mask: torch.Tensor,
        next_question_ids: torch.Tensor,
        next_concept_ids: torch.Tensor,
        next_question_emb: torch.Tensor,
        next_concept_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch, seq_len, hidden_dim = encoded_history.shape
        device = encoded_history.device
        if delta_t is None:
            delta_t = torch.zeros(batch, seq_len, device=device)
        if difficulty is None:
            difficulty = torch.zeros(batch, seq_len, device=device)

        query = self.query_proj(torch.cat([next_question_emb, next_concept_emb], dim=-1))
        key = self.key_proj(encoded_history)
        value = self.value_proj(encoded_history)
        scores = torch.einsum("bd,bld->bl", query, key) / math.sqrt(hidden_dim)

        same_question = question_ids.eq(next_question_ids.unsqueeze(1)) & padding_mask
        same_concept = concept_ids.eq(next_concept_ids.unsqueeze(1)) & padding_mask
        scores = scores + same_question.float() * self.same_question_bias
        scores = scores + same_concept.float() * self.same_concept_bias

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        last_idx = _last_valid_index(padding_mask).unsqueeze(1)
        distance = (last_idx - positions).clamp_min(0).float()
        distance = torch.log1p(distance) / math.log1p(max(self.max_seq_len, 2))
        scores = scores - F.softplus(self.recency_scale) * distance

        reliability_features = torch.stack(
            [
                responses.float().clamp(0, 1),
                1.0 - responses.float().clamp(0, 1),
                delta_t.float().clamp(0, 1),
                difficulty.float().clamp(0, 1),
            ],
            dim=-1,
        )
        scores = scores + torch.log(torch.sigmoid(self.reliability(reliability_features).squeeze(-1)).clamp_min(1e-4))
        scores = scores.masked_fill(~padding_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bl,bld->bd", weights, value)
        gate = self.mix_gate(torch.cat([base_state, context, next_question_emb], dim=-1))
        state = self.norm(base_state + gate * self.dropout(context))
        return {"state": state, "weights": weights, "gate": gate}


class MindTSSemanticHistoryReadout(nn.Module):
    """MindTS-style semantic alignment and condenser for target-aware KT history."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.endogenous_proj = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.cross_view = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.target_query = nn.Linear(hidden_dim * 3, hidden_dim)
        self.history_key = nn.Linear(hidden_dim * 2, hidden_dim)
        self.history_value = nn.Linear(hidden_dim * 2, hidden_dim)
        self.condenser = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.mix_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.reconstruct = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        encoded_history: torch.Tensor,
        base_state: torch.Tensor,
        history_semantic: torch.Tensor,
        target_prompt_state: torch.Tensor,
        next_question_emb: torch.Tensor,
        next_concept_emb: torch.Tensor,
        responses: torch.Tensor,
        delta_t: Optional[torch.Tensor],
        difficulty: Optional[torch.Tensor],
        padding_mask: torch.Tensor,
        semantic_available: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        device = encoded_history.device
        batch, seq_len, hidden_dim = encoded_history.shape
        if delta_t is None:
            delta_t = torch.zeros(batch, seq_len, device=device)
        if difficulty is None:
            difficulty = torch.zeros(batch, seq_len, device=device)
        history_semantic = history_semantic.to(device=device, dtype=encoded_history.dtype)
        history_semantic = history_semantic.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        endo_features = torch.stack(
            [
                responses.float().clamp(0, 1),
                1.0 - responses.float().clamp(0, 1),
                delta_t.float().clamp(0, 1),
                difficulty.float().clamp(0, 1),
            ],
            dim=-1,
        )
        endogenous = self.endogenous_proj(endo_features)
        fused_semantic, _ = self.cross_view(
            endogenous,
            history_semantic,
            history_semantic,
            key_padding_mask=~padding_mask,
            need_weights=False,
        )
        fused_semantic = fused_semantic + endogenous
        item_history = torch.cat([encoded_history, fused_semantic], dim=-1)
        key = self.history_key(item_history)
        value = self.history_value(item_history)
        target = torch.cat([next_question_emb, next_concept_emb, target_prompt_state], dim=-1)
        query = self.target_query(target)
        scores = torch.einsum("bd,bld->bl", query, key) / math.sqrt(hidden_dim)
        condense_input = torch.cat(
            [
                encoded_history,
                fused_semantic,
                torch.abs(encoded_history - fused_semantic),
                query.unsqueeze(1).expand(-1, seq_len, -1),
            ],
            dim=-1,
        )
        mask_prob = self.condenser(condense_input).squeeze(-1)
        semantic_active = semantic_available.to(device=device, dtype=mask_prob.dtype).unsqueeze(-1)
        mask_prob = mask_prob * padding_mask.float() * semantic_active
        scores = scores + torch.log(mask_prob.clamp_min(1e-4))
        scores = scores.masked_fill(~padding_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bl,bld->bd", weights, value)
        gate = self.mix_gate(torch.cat([base_state, context, target_prompt_state, torch.abs(base_state - context)], dim=-1))
        gate = gate * semantic_available.to(device=device, dtype=gate.dtype).unsqueeze(-1)
        state = self.norm(base_state + gate * self.dropout(context))

        time_norm = F.normalize(encoded_history, p=2, dim=-1)
        text_norm = F.normalize(fused_semantic, p=2, dim=-1)
        logits = self.logit_scale.exp().clamp(max=100.0) * torch.einsum("bld,bmd->blm", time_norm, text_norm)
        labels = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        flat_valid = padding_mask.reshape(-1)
        align_time = F.cross_entropy(
            logits.reshape(batch * seq_len, seq_len)[flat_valid],
            labels.reshape(-1)[flat_valid],
        )
        align_text = F.cross_entropy(
            logits.transpose(1, 2).reshape(batch * seq_len, seq_len)[flat_valid],
            labels.reshape(-1)[flat_valid],
        )
        recon = F.mse_loss(
            self.reconstruct(torch.cat([fused_semantic, encoded_history.detach()], dim=-1))[padding_mask],
            encoded_history.detach()[padding_mask],
        )
        compress = (mask_prob[padding_mask].mean() - 0.5).pow(2) if padding_mask.any() else mask_prob.mean() * 0.0
        smooth = (
            (mask_prob[:, 1:] - mask_prob[:, :-1]).pow(2)[padding_mask[:, 1:] & padding_mask[:, :-1]].mean()
            if seq_len > 1 and (padding_mask[:, 1:] & padding_mask[:, :-1]).any()
            else mask_prob.mean() * 0.0
        )
        aux_loss = 0.5 * (align_time + align_text) + 0.2 * recon + 0.05 * compress + 0.02 * smooth
        aux_loss = aux_loss * semantic_available.float().mean()
        return {
            "state": state,
            "weights": weights,
            "gate": gate,
            "mask_prob": mask_prob,
            "aux_loss": aux_loss,
        }


class StudentAwareSemanticAdapter(nn.Module):
    """Turns static semantic evidence into a conservative student-conditioned residual."""

    def __init__(
        self,
        hidden_dim: int,
        context_dim: int = 8,
        dropout: float = 0.1,
        bottleneck_dim: int = 64,
        max_scale: float = 0.25,
        gate_bias: float = -2.0,
        mode: str = "legacy",
    ):
        super().__init__()
        if mode not in {"legacy", "conservative"}:
            raise ValueError("student semantic adapter mode must be 'legacy' or 'conservative'")
        self.mode = mode
        bottleneck_dim = max(1, int(bottleneck_dim))
        self.max_scale = float(max_scale)
        self.scale_logit = nn.Parameter(torch.tensor(-2.0))
        hidden = hidden_dim if mode == "legacy" else bottleneck_dim
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.delta = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden_dim),
            nn.Sigmoid(),
        )
        if mode == "conservative":
            nn.init.constant_(self.gate[-2].bias, gate_bias)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        base_state: torch.Tensor,
        seq_state: torch.Tensor,
        graph_state: torch.Tensor,
        prompt_state: torch.Tensor,
        next_question_emb: torch.Tensor,
        student_context_features: Optional[torch.Tensor],
        semantic_available: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if student_context_features is None:
            student_context_features = torch.zeros(
                base_state.size(0), 8, device=base_state.device, dtype=base_state.dtype
            )
        context = self.context_proj(student_context_features.to(device=base_state.device, dtype=base_state.dtype))
        joint = torch.cat(
            [base_state, seq_state, graph_state, prompt_state, next_question_emb + context],
            dim=-1,
        )
        gate = self.gate(joint)
        gate = gate * semantic_available.to(device=base_state.device, dtype=gate.dtype).unsqueeze(-1)
        delta = self.delta(joint)
        if self.mode == "legacy":
            scale = delta.new_tensor(1.0)
        else:
            scale = torch.sigmoid(self.scale_logit) * self.max_scale
        residual = scale * gate * delta
        state = self.norm(base_state + residual)
        delta_norm = residual.norm(dim=-1)
        base_norm = base_state.norm(dim=-1).clamp_min(1e-6)
        return {
            "state": state,
            "gate": gate,
            "delta_norm": delta_norm,
            "base_norm": base_norm,
            "delta_base_ratio": delta_norm / base_norm,
            "scale": scale.detach().expand(base_state.size(0)),
        }


class EvoTriKT(nn.Module):
    """Behavioral, student knowledge-state graph, and semantic-evidence KT model.

    The model predicts P(r_{t+1}=1 | e_{1:t}, q_{t+1}) instead of a future
    numeric time-series horizon.
    """

    def __init__(
        self,
        num_questions: int,
        num_concepts: int,
        hidden_dim: int = 128,
        prompt_dim: int = 768,
        max_seq_len: int = 200,
        seq_layers: int = 1,
        prompt_layers: int = 1,
        transformer_heads: int = 4,
        alignment_heads: int = 4,
        dropout: float = 0.1,
        semantic_tokens: int = 8,
        graph_adjacency_path: Optional[str] = None,
        use_state_forgetting: bool = True,
        use_rich_graph_evidence: bool = False,
        semantic_fusion_mode: str = "legacy",
        semantic_modality_dropout: float = 0.2,
        question_dropout: float = 0.0,
        folibi_alpha: float = 0.0,
        use_response_interaction: bool = False,
        response_interaction_scale: float = 0.25,
        response_interaction_mode: str = "concept",
        response_interaction_dropout: float = 0.0,
        target_residual_scale: float = 0.5,
        use_target_denoised_pooling: bool = False,
        use_student_semantic_adapter: bool = False,
        student_context_dim: int = 8,
        student_semantic_rank: int = 64,
        student_semantic_dropout: Optional[float] = None,
        student_semantic_max_scale: float = 0.25,
        student_semantic_gate_bias: float = -2.0,
        student_semantic_adapter_mode: str = "legacy",
        use_semantic_advantage_calibration: bool = False,
        semantic_advantage_max_delta: float = 1.0,
        use_mindts_semantic_readout: bool = False,
        use_view_residual_calibration: bool = False,
        view_residual_max_delta: float = 0.6,
    ):
        super().__init__()
        self.supports_multi_concept = True
        self.hidden_dim = hidden_dim
        self.question_dropout = float(question_dropout)
        self.use_target_denoised_pooling = bool(use_target_denoised_pooling)
        self.use_student_semantic_adapter = bool(use_student_semantic_adapter)
        self.use_semantic_advantage_calibration = bool(use_semantic_advantage_calibration)
        self.use_mindts_semantic_readout = bool(use_mindts_semantic_readout)
        self.use_view_residual_calibration = bool(use_view_residual_calibration)
        self.use_rich_graph_evidence = bool(use_rich_graph_evidence)
        if semantic_fusion_mode not in {"legacy", "student_gated"}:
            raise ValueError(
                "semantic_fusion_mode must be one of {'legacy', 'student_gated'}"
            )
        self.semantic_fusion_mode = semantic_fusion_mode
        adjacency = self._load_adjacency(graph_adjacency_path, num_questions, num_concepts)
        self.sequence_encoder = KnowledgeInteractionSequenceEncoder(
            num_questions=num_questions,
            num_concepts=num_concepts,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            num_layers=seq_layers,
            num_heads=transformer_heads,
            dropout=dropout,
            question_dropout=question_dropout,
            folibi_alpha=folibi_alpha,
            use_response_interaction=use_response_interaction,
            response_interaction_scale=response_interaction_scale,
            response_interaction_mode=response_interaction_mode,
            response_interaction_dropout=response_interaction_dropout,
        )
        # Attribute names are retained for compatibility with existing checkpoints.
        self.graph_builder = CausalStudentKnowledgeStateUpdater(
            num_atomic_concepts=adjacency["num_atomic_concepts"],
            hidden_dim=hidden_dim,
            question_to_atomic_indices=adjacency["question_to_atomic_indices"],
            concept_to_atomic_indices=adjacency["concept_to_atomic_indices"],
            cooc_norm=adjacency["cooc_norm"],
            use_state_forgetting=use_state_forgetting,
            use_rich_graph_evidence=use_rich_graph_evidence,
        )
        self.graph_encoder = StudentKnowledgeStateGraphReadout(
            hidden_dim=hidden_dim,
            num_atomic_concepts=adjacency["num_atomic_concepts"],
            question_to_atomic_indices=adjacency["question_to_atomic_indices"],
            concept_to_atomic_indices=adjacency["concept_to_atomic_indices"],
        )
        self.prompt_encoder = SemanticPromptEncoder(
            prompt_dim=prompt_dim,
            hidden_dim=hidden_dim,
            num_layers=prompt_layers,
            num_heads=transformer_heads,
            dropout=dropout,
            semantic_tokens=semantic_tokens,
        )
        self.prompt_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.temporal_structural_gate = StudentStateAwareTemporalStructuralGate(hidden_dim=hidden_dim)
        self.target_adapter = ConceptAnchoredTargetAdapter(
            hidden_dim=hidden_dim,
            dropout=dropout,
            residual_scale=target_residual_scale,
        )
        self.target_denoised_pool = TargetAwareDenoisedHistoryPooling(
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.mindts_semantic_readout = MindTSSemanticHistoryReadout(
            hidden_dim=hidden_dim,
            num_heads=transformer_heads,
            dropout=dropout,
        )
        self.student_semantic_adapter = StudentAwareSemanticAdapter(
            hidden_dim=hidden_dim,
            context_dim=student_context_dim,
            dropout=dropout if student_semantic_dropout is None else student_semantic_dropout,
            bottleneck_dim=student_semantic_rank,
            max_scale=student_semantic_max_scale,
            gate_bias=student_semantic_gate_bias,
            mode=student_semantic_adapter_mode,
        )
        self.semantic_advantage = SemanticAdvantageCalibration(
            hidden_dim=hidden_dim,
            max_delta=semantic_advantage_max_delta,
            dropout=dropout,
        )
        self.view_residual_calibration = TriViewResidualLogitCalibration(
            hidden_dim=hidden_dim,
            max_delta=view_residual_max_delta,
            dropout=dropout,
        )
        if semantic_fusion_mode == "legacy":
            self.causal_alignment = CausalMultiHeadCrossModalAlignment(
                hidden_dim=hidden_dim,
                num_alignment_heads=alignment_heads,
                attn_heads=transformer_heads,
                dropout=dropout,
            )
            self.head_router = KnowledgeAdaptiveHeadRouter(
                hidden_dim=hidden_dim, num_alignment_heads=alignment_heads
            )
            self.dynamic_residual = EvidencePreservingDynamicResidualFusion(
                hidden_dim=hidden_dim
            )
            self.student_semantic_fusion = None
        else:
            self.causal_alignment = None
            self.head_router = None
            self.dynamic_residual = None
            self.student_semantic_fusion = StudentStateGatedSemanticFusion(
                hidden_dim=hidden_dim,
                num_heads=transformer_heads,
                dropout=dropout,
                modality_dropout=semantic_modality_dropout,
            )
        self.prediction_head = KnowledgeStatePredictionHead(hidden_dim=hidden_dim)
        self.view_predictors = nn.ModuleDict(
            {
                "sequence": nn.Linear(hidden_dim, 1),
                "graph": nn.Linear(hidden_dim, 1),
                "prompt": nn.Linear(hidden_dim, 1),
            }
        )

    @staticmethod
    def _load_adjacency(
        path: Optional[str], num_questions: int, num_concepts: int
    ) -> Dict[str, torch.Tensor]:
        if path is not None:
            data = torch.load(path, map_location="cpu")
            required = {
                "question_to_atomic_indices",
                "concept_to_atomic_indices",
                "cooc_norm",
                "num_atomic_concepts",
            }
            missing = required.difference(data)
            if missing:
                raise KeyError(f"Graph adjacency is missing keys: {sorted(missing)}")
            return {
                "question_to_atomic_indices": data["question_to_atomic_indices"].long(),
                "concept_to_atomic_indices": data["concept_to_atomic_indices"].long(),
                "cooc_norm": data["cooc_norm"].float(),
                "num_atomic_concepts": int(data["num_atomic_concepts"]),
            }

        # Identity fallback keeps old checkpoints/scripts runnable, but strict tri-modal
        # training should always provide a dataset-specific adjacency file.
        num_atomic = max(num_concepts - 2, 1)
        q2a = torch.full((num_questions, 1), -1, dtype=torch.long)
        c2a = torch.full((num_concepts, 1), -1, dtype=torch.long)
        if num_concepts > 2:
            c2a[2:, 0] = torch.arange(num_concepts - 2)
        return {
            "question_to_atomic_indices": q2a,
            "concept_to_atomic_indices": c2a,
            "cooc_norm": torch.eye(num_atomic, dtype=torch.float32),
            "num_atomic_concepts": num_atomic,
        }

    def forward(
        self,
        question_ids: torch.Tensor,
        concept_ids: torch.Tensor,
        responses: torch.Tensor,
        next_question_ids: torch.Tensor,
        next_concept_ids: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
        retention_age: Optional[torch.Tensor] = None,
        difficulty: Optional[torch.Tensor] = None,
        prompt_embeddings: Optional[torch.Tensor] = None,
        history_prompt_embeddings: Optional[torch.Tensor] = None,
        student_context_features: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_aux: bool = True,
        concept_set_ids: Optional[torch.Tensor] = None,
        concept_set_mask: Optional[torch.Tensor] = None,
        next_concept_set_ids: Optional[torch.Tensor] = None,
        next_concept_set_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if padding_mask is None:
            padding_mask = question_ids.ne(0)
        batch_size = question_ids.size(0)
        device = question_ids.device

        encoded_history = self.sequence_encoder(
            question_ids,
            concept_ids,
            responses,
            delta_t,
            difficulty,
            padding_mask,
            return_sequence=True,
            concept_set_ids=concept_set_ids,
            concept_set_mask=concept_set_mask,
        )
        seq_state = _gather_last(encoded_history, padding_mask)
        graph_pack = self.graph_builder(
            question_ids,
            concept_ids,
            responses,
            delta_t,
            difficulty,
            padding_mask,
            retention_age=retention_age,
            event_evidence_state=encoded_history if self.use_rich_graph_evidence else None,
        )
        raw_next_question_emb = self.sequence_encoder.question_emb(next_question_ids.clamp_min(0))
        next_concept_emb = self.sequence_encoder.concept_emb(next_concept_ids.clamp_min(0))
        if next_concept_set_ids is not None:
            set_ids = next_concept_set_ids.clamp(
                0, self.sequence_encoder.concept_emb.num_embeddings - 1
            )
            set_emb = self.sequence_encoder.concept_emb(set_ids)
            if next_concept_set_mask is None:
                next_concept_set_mask = next_concept_set_ids.gt(0)
            weights = next_concept_set_mask.to(set_emb.dtype).unsqueeze(-1)
            next_concept_emb = (set_emb * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
        next_question_emb = self.target_adapter(
            raw_next_question_emb,
            next_concept_emb,
            question_dropout=self.question_dropout,
        )
        denoised_pool = None
        if self.use_target_denoised_pooling:
            denoised_pool = self.target_denoised_pool(
                encoded_history=encoded_history,
                base_state=seq_state,
                question_ids=question_ids,
                concept_ids=concept_ids,
                responses=responses,
                delta_t=delta_t,
                difficulty=difficulty,
                padding_mask=padding_mask,
                next_question_ids=next_question_ids,
                next_concept_ids=next_concept_ids,
                next_question_emb=next_question_emb,
                next_concept_emb=next_concept_emb,
            )
            seq_state = denoised_pool["state"]
        graph_state = self.graph_encoder(
            graph_pack["node_state"], next_question_ids, next_concept_ids, next_concept_emb
        )
        target_prompt_tokens = self.prompt_encoder(prompt_embeddings, batch_size, device)
        history_semantic_tokens = None
        if history_prompt_embeddings is not None:
            history_prompt_embeddings = history_prompt_embeddings.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            history_prompt_mask = history_prompt_embeddings.abs().sum(dim=-1).gt(0)
            history_semantic_tokens = self.prompt_encoder.prompt_proj(history_prompt_embeddings)
            history_semantic_tokens = history_semantic_tokens.masked_fill(
                ~history_prompt_mask.unsqueeze(-1), 0.0
            )
            prompt_tokens = torch.cat([history_semantic_tokens, target_prompt_tokens], dim=1)
            prompt_mask = torch.cat(
                [
                    history_prompt_mask,
                    torch.ones(
                        batch_size,
                        target_prompt_tokens.size(1),
                        device=device,
                        dtype=torch.bool,
                    ),
                ],
                dim=1,
            )
        else:
            prompt_tokens = target_prompt_tokens
            prompt_mask = torch.ones(
                batch_size, prompt_tokens.size(1), device=device, dtype=torch.bool
            )
        prompt_scores = self.prompt_pool(prompt_tokens).squeeze(-1)
        prompt_scores = prompt_scores.masked_fill(~prompt_mask, -1e4)
        prompt_state = torch.sum(
            torch.softmax(prompt_scores, dim=1).unsqueeze(-1) * prompt_tokens, dim=1
        )
        if prompt_embeddings is None:
            semantic_available = torch.zeros(
                batch_size, device=device, dtype=torch.bool
            )
        else:
            semantic_available = prompt_embeddings.to(device).flatten(start_dim=1).abs().sum(dim=1).gt(0)

        mindts_readout = None
        if self.use_mindts_semantic_readout and history_semantic_tokens is not None:
            mindts_readout = self.mindts_semantic_readout(
                encoded_history=encoded_history,
                base_state=seq_state,
                history_semantic=history_semantic_tokens,
                target_prompt_state=prompt_state,
                next_question_emb=next_question_emb,
                next_concept_emb=next_concept_emb,
                responses=responses,
                delta_t=delta_t,
                difficulty=difficulty,
                padding_mask=padding_mask,
                semantic_available=semantic_available,
            )
            seq_state = mindts_readout["state"]

        last_delta_t = torch.zeros(batch_size, device=device) if delta_t is None else _gather_last(delta_t.float(), padding_mask)
        gated = self.temporal_structural_gate(
            seq_state=seq_state,
            graph_state=graph_state,
            next_question_emb=next_question_emb,
            next_concept_emb=next_concept_emb,
            last_delta_t=last_delta_t,
            graph_confidence=graph_pack["confidence"],
            history_len=graph_pack["history_len"],
        )
        base_state = gated["state"]
        sequence_logits = self.view_predictors["sequence"](seq_state).squeeze(-1)
        graph_logits = self.view_predictors["graph"](graph_state).squeeze(-1)
        prompt_logits = self.view_predictors["prompt"](prompt_state).squeeze(-1)
        p_seq = torch.sigmoid(sequence_logits)
        p_graph = torch.sigmoid(graph_logits)
        p_prompt = torch.sigmoid(prompt_logits)
        if prompt_embeddings is None or self.semantic_fusion_mode == "student_gated":
            conflict = _binary_js_divergence(p_seq, p_graph)
        else:
            conflict = (
                _binary_js_divergence(p_seq, p_graph)
                + _binary_js_divergence(p_seq, p_prompt)
                + _binary_js_divergence(p_graph, p_prompt)
            ) / 3.0

        if self.semantic_fusion_mode == "legacy":
            head_outputs = self.causal_alignment(base_state, prompt_tokens, prompt_mask)
            routed = self.head_router(head_outputs, base_state)
            if prompt_embeddings is None:
                routed["state"] = base_state
            residual = self.dynamic_residual(
                base_state, routed["state"], next_question_emb, conflict
            )
            semantic_weights = routed["weights"]
        else:
            residual = self.student_semantic_fusion(
                base_state=base_state,
                prompt_tokens=prompt_tokens,
                next_question_emb=next_question_emb,
                prompt_mask=prompt_mask,
                semantic_available=semantic_available,
            )
            semantic_weights = residual["weights"]
        adapted = None
        advantage = None
        final_state = residual["state"]
        base_prediction_state = final_state
        base_logits = self.prediction_head(base_prediction_state, next_question_emb, next_concept_emb)
        semantic_logits = base_logits
        if self.use_student_semantic_adapter:
            adapted = self.student_semantic_adapter(
                base_state=final_state,
                seq_state=seq_state,
                graph_state=graph_state,
                prompt_state=prompt_state,
                next_question_emb=next_question_emb,
                student_context_features=student_context_features,
                semantic_available=semantic_available,
            )
            if self.use_semantic_advantage_calibration:
                semantic_logits = self.prediction_head(
                    adapted["state"], next_question_emb, next_concept_emb
                )
                advantage = self.semantic_advantage(
                    base_state=base_prediction_state,
                    semantic_state=adapted["state"],
                    seq_state=seq_state,
                    graph_state=graph_state,
                    prompt_state=prompt_state,
                    base_logit=base_logits,
                    semantic_logit=semantic_logits,
                    p_sequence=p_seq,
                    p_graph=p_graph,
                    p_prompt=p_prompt,
                    conflict=conflict,
                    graph_confidence=graph_pack["confidence"],
                    semantic_available=semantic_available,
                )
                logits = advantage["logits"]
            else:
                final_state = adapted["state"]
                logits = self.prediction_head(final_state, next_question_emb, next_concept_emb)
        else:
            logits = base_logits
        view_residual = None
        if self.use_view_residual_calibration:
            view_residual = self.view_residual_calibration(
                base_state=base_prediction_state,
                seq_state=seq_state,
                graph_state=graph_state,
                prompt_state=prompt_state,
                next_question_emb=next_question_emb,
                base_logit=logits,
                sequence_logit=sequence_logits,
                graph_logit=graph_logits,
                prompt_logit=prompt_logits,
                conflict=conflict,
                graph_confidence=graph_pack["confidence"],
                semantic_available=semantic_available,
            )
            logits = view_residual["logits"]
        probs = torch.sigmoid(logits)

        output = {"logits": logits, "probs": probs}
        if return_aux:
            output.update(
                {
                    "seq_state": seq_state,
                    "graph_state": graph_state,
                    "prompt_state": prompt_state,
                    "temporal_structural_gate": gated["gate"],
                    "kar_gate": gated["gate"],
                    "head_weights": semantic_weights,
                    "residual_gamma": residual["gamma"],
                    "scaef_alpha": residual["gamma"],
                    "view_conflict": conflict,
                    "graph_confidence": graph_pack["confidence"],
                    "semantic_available": semantic_available,
                    "graph_visit_count": graph_pack["visit_count"],
                    "mean_retention": graph_pack["mean_retention"],
                    "mindts_semantic_aux_loss": (
                        mindts_readout["aux_loss"].expand(batch_size)
                        if mindts_readout is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "mindts_semantic_gate": (
                        mindts_readout["gate"].mean(dim=1)
                        if mindts_readout is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "mindts_semantic_mask": (
                        mindts_readout["mask_prob"].mean(dim=1)
                        if mindts_readout is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "denoised_history_gate": (
                        denoised_pool["gate"].mean(dim=1)
                        if denoised_pool is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "student_semantic_gate": (
                        adapted["gate"].mean(dim=1)
                        if adapted is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "student_semantic_gate_std": (
                        adapted["gate"].std(dim=1)
                        if adapted is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "student_semantic_delta_norm": (
                        adapted["delta_norm"]
                        if adapted is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "student_semantic_base_norm": (
                        adapted["base_norm"]
                        if adapted is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "student_semantic_delta_base_ratio": (
                        adapted["delta_base_ratio"]
                        if adapted is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "student_semantic_scale": (
                        adapted["scale"]
                        if adapted is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "semantic_advantage_lambda": (
                        advantage["lambda"]
                        if advantage is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "semantic_correction_mean": (
                        advantage["bounded_correction"]
                        if advantage is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "semantic_correction_abs": (
                        advantage["bounded_correction"].abs()
                        if advantage is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "view_residual_gate": (
                        view_residual["gates"].mean(dim=1)
                        if view_residual is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "view_residual_correction_abs": (
                        view_residual["correction"].abs()
                        if view_residual is not None
                        else torch.zeros(batch_size, device=device)
                    ),
                    "base_logits": base_logits,
                    "semantic_logits": semantic_logits,
                    "base_probs": torch.sigmoid(base_logits),
                    "semantic_probs": torch.sigmoid(semantic_logits),
                    "p_sequence": p_seq,
                    "p_graph": p_graph,
                    "p_prompt": p_prompt,
                    "sequence_logits": sequence_logits,
                    "graph_logits": graph_logits,
                    "prompt_logits": prompt_logits,
                }
            )
        return output
