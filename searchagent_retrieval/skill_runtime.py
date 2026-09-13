"""Lightweight runtime Skill catalog, query builder, and mount telemetry."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BGE_LARGE_ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    description: str
    path: Path
    text: str
    stages: tuple[str, ...]
    trigger_text: str
    exclusion_text: str


class SkillCatalog:
    """Discover Skills from SKILL.md and retrieve candidates by stage/query.

    The catalog deliberately stores only routing metadata; full instructions
    are loaded lazily by ``load_skill`` in the generation tool loop.
    """

    def __init__(
        self,
        root: Path | None = None,
        embed_model_path: Path | None = None,
        embedding_cache_dir: Path | None = None,
        allow_candidates: bool = False,
    ) -> None:
        self.root = (root or Path(__file__).resolve().parents[1] / "skill_evolution" / "skills").resolve()
        self.allow_candidates = bool(allow_candidates)
        self.skills = self._discover()
        self.embed_model_path = Path(embed_model_path).resolve() if embed_model_path is not None else None
        self.embedding_cache_dir = (
            Path(embedding_cache_dir).resolve()
            if embedding_cache_dir is not None
            else self.root.parent / "registry" / "embedding_cache"
        )
        self._embedder = None
        self._skill_vectors: dict[str, Any] | None = None

    def _ensure_embedder(self) -> Any:
        if self.embed_model_path is None:
            return None
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(str(self.embed_model_path))
        return self._embedder

    def _encode_query(self, query: str) -> Any:
        try:
            embedder = self._ensure_embedder()
            if embedder is None:
                return None
            import numpy as np
            text = str(query)
            model_label = str(self.embed_model_path).lower()
            if "bge" in model_label and "zh" in model_label:
                text = BGE_LARGE_ZH_QUERY_INSTRUCTION + text
            return np.asarray(
                embedder.encode([text], normalize_embeddings=True, show_progress_bar=False),
                dtype="float32",
            )[0]
        except Exception:
            return None

    def _encode_skill_documents(self, texts: list[str]) -> Any:
        embedder = self._ensure_embedder()
        if embedder is None:
            return None
        import numpy as np
        return np.asarray(
            embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype="float32",
        )

    @staticmethod
    def _searchable_text(skill: SkillSpec) -> str:
        return f"{skill.skill_id} {skill.description} {skill.trigger_text} {skill.exclusion_text} {skill.text[:4000]}".lower()

    @staticmethod
    def _skill_hash(skill: SkillSpec) -> str:
        return hashlib.sha256(skill.text.encode()).hexdigest()[:12]

    def _embedding_index_paths(self) -> tuple[Path, Path]:
        model_label = str(self.embed_model_path or "keyword")
        model_key = hashlib.sha256(model_label.encode()).hexdigest()[:12]
        return (
            self.embedding_cache_dir / f"skills_{model_key}.json",
            self.embedding_cache_dir / f"skills_{model_key}.npz",
        )

    def ensure_embedding_index(self, *, force: bool = False) -> dict[str, Any]:
        """Load or build persistent Skill document vectors.

        Skill passages are encoded only when the model identity or a SKILL.md
        hash changes. Runtime retrieval then encodes only the query.
        """
        if self.embed_model_path is None:
            return {"mode": "keyword", "rebuilt": False, "skill_count": len(self.skills)}
        if self._skill_vectors is not None and not force:
            return {"mode": "embedding", "rebuilt": False, "skill_count": len(self._skill_vectors)}

        import numpy as np

        manifest_path, vectors_path = self._embedding_index_paths()
        skill_ids = [skill.skill_id for skill in self.skills]
        skill_hashes = {skill.skill_id: self._skill_hash(skill) for skill in self.skills}
        expected = {
            "model": str(self.embed_model_path),
            "skill_ids": skill_ids,
            "skill_hashes": skill_hashes,
        }
        if not force and manifest_path.is_file() and vectors_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if all(manifest.get(key) == value for key, value in expected.items()):
                    with np.load(vectors_path, allow_pickle=False) as archive:
                        matrix = np.asarray(archive["vectors"], dtype="float32")
                    if matrix.ndim == 2 and matrix.shape[0] == len(skill_ids):
                        self._skill_vectors = {
                            skill_id: matrix[index]
                            for index, skill_id in enumerate(skill_ids)
                        }
                        return {"mode": "embedding", "rebuilt": False, "skill_count": len(skill_ids)}
            except Exception:
                pass

        texts = [self._searchable_text(skill) for skill in self.skills]
        matrix = self._encode_skill_documents(texts)
        if matrix is None:
            return {"mode": "keyword", "rebuilt": False, "skill_count": len(self.skills)}
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        temp_suffix = f".tmp.{os.getpid()}"
        temp_vectors = vectors_path.with_name(vectors_path.name + temp_suffix)
        temp_manifest = manifest_path.with_name(manifest_path.name + temp_suffix)
        with temp_vectors.open("wb") as handle:
            np.savez_compressed(handle, vectors=matrix)
        manifest = {**expected, "dimension": int(matrix.shape[1]), "format_version": 1}
        temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp_vectors, vectors_path)
        os.replace(temp_manifest, manifest_path)
        self._skill_vectors = {
            skill_id: matrix[index]
            for index, skill_id in enumerate(skill_ids)
        }
        return {"mode": "embedding", "rebuilt": True, "skill_count": len(skill_ids)}

    def _discover(self) -> list[SkillSpec]:
        result: list[SkillSpec] = []
        configured_allowlist = {
            value.strip()
            for value in os.environ.get("DYNAMIC_SKILL_ALLOWLIST", "").split(",")
            if value.strip()
        }
        registry_path = self.root.parent / "registry" / "registry.json"
        registry_status: dict[str, str] = {}
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry_status = {
                str(row.get("skill_id")): str(row.get("status") or "")
                for row in registry.get("skills") or [] if isinstance(row, dict)
            }
        except (OSError, json.JSONDecodeError):
            pass
        allow_candidates = self.allow_candidates or os.environ.get("DYNAMIC_SKILL_ALLOW_CANDIDATES", "").strip().lower() in {"1", "true", "yes", "on"}
        if not self.root.is_dir():
            return result
        for path in sorted(self.root.glob("*/SKILL.md")):
            if configured_allowlist and path.parent.name not in configured_allowlist:
                continue
            # Normal runtime is fail-closed: only A/B-promoted Skills are
            # discoverable. An explicit allowlist is the source-error A/B seam.
            status = registry_status.get(path.parent.name)
            if status != "active" and not allow_candidates and path.parent.name not in configured_allowlist:
                continue
            text = path.read_text(encoding="utf-8")
            front: dict[str, str] = {}
            metadata: dict[str, str] = {}
            in_frontmatter = False
            in_metadata = False
            for line in text.splitlines():
                if line.strip() == "---":
                    if in_frontmatter:
                        break
                    in_frontmatter = True
                    continue
                if not in_frontmatter or ":" not in line or line.lstrip().startswith("#"):
                    continue
                indent = len(line) - len(line.lstrip())
                key, value = line.strip().split(":", 1)
                key = key.strip().lower()
                value = value.strip().strip("'\"")
                if indent == 0:
                    in_metadata = key == "metadata"
                    if key != "metadata":
                        front[key] = value
                elif in_metadata:
                    metadata[key] = value
            description = front.get("description") or front.get("name") or path.parent.name
            stages = tuple(x.strip() for x in re.split(r"[,|]", metadata.get("call_stage", "evidence_review|answer_context|generate")) if x.strip())
            if stages and all(stage in {"merged_candidate", "retired_candidate", "disabled"} for stage in stages):
                continue
            trigger = metadata.get("when_to_use", "")
            exclusion = metadata.get("when_not_to_use", "")
            result.append(SkillSpec(path.parent.name, description, path, text, stages, trigger, exclusion))
        return result

    @staticmethod
    def build_query(*, stage: str, goal: str = "", signals: Iterable[str] = (), gaps: Iterable[str] = ()) -> str:
        parts = [f"stage={stage}", f"goal={goal}"]
        signal_text = " ".join(str(x) for x in signals if str(x).strip())
        gap_text = " ".join(str(x) for x in gaps if str(x).strip())
        if signal_text:
            parts.append(f"signals={signal_text}")
        if gap_text:
            parts.append(f"gaps={gap_text}")
        return " ".join(parts)

    @staticmethod
    def lineage_anomaly_signals(
        *,
        answer_memory: dict[str, Any],
        answer_context: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Derive entity-free Skill routing signals from downstream artifacts.

        Concrete drugs, mutations, diseases, and claim prose remain in the
        stage payload.  Skill retrieval sees only structural anomalies so the
        same Skill can generalize across clinical entities.
        """
        signals: list[str] = []
        gaps: list[str] = []

        claims = [row for row in (answer_memory.get("claims") or []) if isinstance(row, dict)]
        claims_by_id = {str(row.get("claim_id") or ""): row for row in claims if row.get("claim_id")}
        expected_evidence_ids: set[str] = set()
        for claim in claims:
            if str(claim.get("status") or "") in {"rejected_by_safety_gate", "rejected", "discarded"}:
                continue
            scopes = {str(value).strip() for value in (claim.get("evidence_scopes") or []) if str(value).strip()}
            if len(scopes) > 1 and "claim_scope_collision" not in signals:
                signals.append("claim_scope_collision")
                gaps.append("split_incompatible_claim_scopes")
            for field in (
                "direct_support_chunk_ids",
                "partial_support_chunk_ids",
                "analog_support_chunk_ids",
                "contradicting_chunk_ids",
            ):
                expected_evidence_ids.update(str(value) for value in (claim.get(field) or []) if str(value))

        finding_rows = [
            row
            for field in ("key_findings", "partial_or_analog_findings")
            for row in (answer_context.get(field) or [])
            if isinstance(row, dict)
        ]
        retained_evidence_ids = {str(value) for value in (answer_context.get("evidence_ids") or []) if str(value)}
        for finding in finding_rows:
            retained_evidence_ids.update(str(value) for value in (finding.get("evidence_ids") or []) if str(value))
            claim_id = str(finding.get("claim_id") or "")
            evidence_ids = [str(value) for value in (finding.get("evidence_ids") or []) if str(value)]
            if not claim_id or not evidence_ids:
                if "evidence_derived_conclusion_without_binding" not in signals:
                    signals.append("evidence_derived_conclusion_without_binding")
                    gaps.append("restore_claim_and_evidence_binding")
                continue
            source_claim = claims_by_id.get(claim_id) or {}
            expected_level = str(source_claim.get("support_level") or "")
            context_level = "direct" if finding in (answer_context.get("key_findings") or []) else str(finding.get("support_level") or "")
            if expected_level and context_level and expected_level != context_level:
                if "evidence_state_flip" not in signals:
                    signals.append("evidence_state_flip")
                    gaps.append("restore_original_evidence_state")

        if expected_evidence_ids - retained_evidence_ids:
            signals.append("unexplained_evidence_drop")
            gaps.append("account_for_dropped_evidence_ids")
        if answer_context.get("prohibited_attributions"):
            signals.append("cross_intervention_attribution")
            gaps.append("preserve_intervention_attribution_boundary")

        return list(dict.fromkeys(signals)), list(dict.fromkeys(gaps))

    def mount_for_signals(
        self,
        *,
        stage: str,
        goal: str,
        signals: Iterable[str],
        gaps: Iterable[str] = (),
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Retrieve Skills only when an observable structural signal exists."""
        signal_rows = [str(value).strip() for value in signals if str(value).strip()]
        gap_rows = [str(value).strip() for value in gaps if str(value).strip()]
        query = self.build_query(stage=stage, goal=goal, signals=signal_rows, gaps=gap_rows)
        if not signal_rows:
            return {
                "stage": stage,
                "query": query,
                "routing_signals": [],
                "candidate_skills": [],
                "selected_skill_ids": [],
                "loaded_skill_ids": [],
                "validation": "not_needed",
            }
        state = self.mount(query=query, stage=stage, top_k=top_k)
        state["routing_signals"] = signal_rows
        return state

    def retrieve(self, query: str, *, stage: str, top_k: int = 5) -> list[dict[str, Any]]:
        tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
        rows = []
        eligible = [skill for skill in self.skills if stage in skill.stages or "*" in skill.stages]
        vectors = None
        query_vector = None
        if eligible and self.embed_model_path is not None:
            try:
                self.ensure_embedding_index()
                vectors = self._skill_vectors
                query_vector = self._encode_query(query)
            except Exception:
                vectors = None
                query_vector = None
        for skill in self.skills:
            if stage not in skill.stages and "*" not in skill.stages:
                continue
            searchable = self._searchable_text(skill)
            matched = sum(1 for token in tokens if token and token in searchable)
            score = matched / max(1, len(tokens))
            if vectors is not None and query_vector is not None and skill.skill_id in vectors:
                try:
                    score = float(query_vector @ vectors[skill.skill_id])
                except Exception:
                    pass
            rows.append({"skill_id": skill.skill_id, "description": skill.description, "when_to_use": skill.trigger_text, "when_not_to_use": skill.exclusion_text, "score": round(score, 4), "embedding_text": searchable[:4000], "skill_hash": self._skill_hash(skill)})
        rows.sort(key=lambda x: (-x["score"], x["skill_id"]))
        return rows[:top_k]

    def mount(self, *, query: str, stage: str, top_k: int = 3) -> dict[str, Any]:
        candidates = self.retrieve(query, stage=stage, top_k=top_k)
        return {"stage": stage, "query": query, "candidate_skills": candidates, "selected_skill_ids": [], "loaded_skill_ids": [], "validation": "pending"}

    @staticmethod
    def record_selection(state: dict[str, Any], selected_ids: Iterable[str], loaded_ids: Iterable[str]) -> dict[str, Any]:
        selected = [str(x) for x in selected_ids if str(x)]
        loaded = [str(x) for x in loaded_ids if str(x)]
        allowed = {str(x.get("skill_id")) for x in state.get("candidate_skills") or []}
        state["selected_skill_ids"] = [x for x in selected if x in allowed]
        state["loaded_skill_ids"] = [x for x in loaded if x in state["selected_skill_ids"]]
        state["validation"] = "passed" if set(state["selected_skill_ids"]).issubset(set(state["loaded_skill_ids"])) else "failed"
        return state
