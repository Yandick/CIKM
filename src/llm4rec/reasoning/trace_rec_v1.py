"""TRACE-Rec V1 graph schema parsing and validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class GraphNodeType(str, Enum):
    PREFERENCE_STATE = "preference_state"
    CANDIDATE_EVIDENCE = "candidate_evidence"
    DECISION = "decision"


class GraphEdgeType(str, Enum):
    SUPPORTS = "supports"
    CONFLICTS = "conflicts"
    SELECTED = "selected"


class PreferencePolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class PreferenceHorizon(str, Enum):
    RECENT = "recent"
    PERSISTENT = "persistent"


class PreferenceSource(str, Enum):
    HISTORY = "history"
    CONTEXT = "context"
    CONSTRAINT = "constraint"


@dataclass(frozen=True)
class TraceRecV1Constraints:
    """Configurable schema constraints loaded from the prompt YAML."""

    candidate_conditioned: bool = True
    require_evidence_grounding: bool = True
    max_preference_state_nodes: int = 3
    max_candidate_evidence_nodes: int = 5
    max_signed_edges_per_candidate_evidence: int = 2
    exact_selected_edges: int = 1
    strict_single_pointer_mode: bool = True


@dataclass(frozen=True)
class TraceRecV1SchemaConfig:
    """Loaded TRACE-Rec V1 prompt schema configuration."""

    name: str
    constraints: TraceRecV1Constraints


@dataclass(frozen=True)
class GraphMetadata:
    """Graph-level metadata carried with every TRACE-Rec example."""

    task: str
    user_id: str
    candidate_ids: tuple[str, ...]
    context: Any | None = None

    def evidence_registry(self) -> set[str]:
        """Return the global evidence-ref registry when the metadata carries one."""

        if not isinstance(self.context, dict):
            return set()
        refs = set(str(item) for item in self.context.get("available_evidence_refs", []))
        refs.update(str(item) for item in self.context.get("available_history_refs", []))
        refs.update(str(item) for item in self.context.get("available_context_refs", []))
        refs.update(str(item) for item in self.context.get("available_constraint_refs", []))
        return refs

    def feature_registry(self, candidate_id: str) -> set[str]:
        """Return the candidate-scoped feature-ref registry when available."""

        if not isinstance(self.context, dict):
            return set()

        refs = set(str(item) for item in self.context.get("available_feature_refs", []))
        by_candidate = self.context.get("available_feature_refs_by_candidate", {})
        if isinstance(by_candidate, dict):
            refs.update(str(item) for item in by_candidate.get(candidate_id, []))
        return refs


@dataclass(frozen=True)
class PreferenceStateNode:
    """User-side signal grounded in history, context, or constraints."""

    id: str
    summary: str
    polarity: PreferencePolarity
    horizon: PreferenceHorizon
    evidence_refs: tuple[str, ...]
    source: PreferenceSource
    type: GraphNodeType = GraphNodeType.PREFERENCE_STATE


@dataclass(frozen=True)
class CandidateEvidenceNode:
    """Candidate-conditioned evidence unit for one candidate item."""

    id: str
    candidate_id: str
    feature_refs: tuple[str, ...]
    summary: str
    rank_prior: float | None = None
    retrieval_source: str | None = None
    type: GraphNodeType = GraphNodeType.CANDIDATE_EVIDENCE


@dataclass(frozen=True)
class DecisionNode:
    """Final recommendation node."""

    id: str
    selected_item_id: str
    type: GraphNodeType = GraphNodeType.DECISION


TraceRecNode = PreferenceStateNode | CandidateEvidenceNode | DecisionNode


@dataclass(frozen=True)
class GraphEdge:
    """Directed relation between TRACE-Rec nodes."""

    source: str
    target: str
    type: GraphEdgeType
    strength: str | None = None
    reason_span: str | None = None


@dataclass(frozen=True)
class TraceRecGraph:
    """Typed TRACE-Rec graph object."""

    metadata: GraphMetadata
    nodes: tuple[TraceRecNode, ...]
    edges: tuple[GraphEdge, ...]

    def node_by_id(self) -> dict[str, TraceRecNode]:
        """Return a node lookup table keyed by node id."""

        return {node.id: node for node in self.nodes}


@dataclass(frozen=True)
class ValidationIssue:
    """One validation issue emitted by the parser or validator."""

    code: str
    message: str
    path: str


@dataclass
class ValidationResult:
    """Validation result wrapper used by graph and eval validators."""

    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Whether the validation produced no issues."""

        return not self.issues

    def add(self, code: str, message: str, path: str) -> None:
        """Append a new validation issue."""

        self.issues.append(ValidationIssue(code=code, message=message, path=path))


class TraceRecGraphValidationError(ValueError):
    """Raised when a TRACE-Rec graph fails strict parsing."""

    def __init__(self, result: ValidationResult) -> None:
        self.result = result
        message = "; ".join(f"{issue.path}: {issue.message}" for issue in result.issues)
        super().__init__(message)


def load_trace_rec_v1_schema_config(config_path: str | Path) -> TraceRecV1SchemaConfig:
    """Load the TRACE-Rec V1 schema config from the prompt YAML file."""

    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    constraints = data.get("constraints", {})
    return TraceRecV1SchemaConfig(
        name=str(data.get("name", "trace_rec_v1")),
        constraints=TraceRecV1Constraints(
            candidate_conditioned=bool(constraints.get("candidate_conditioned", True)),
            require_evidence_grounding=bool(
                constraints.get("require_evidence_grounding", True)
            ),
            max_preference_state_nodes=int(constraints.get("max_preference_state_nodes", 3)),
            max_candidate_evidence_nodes=int(
                constraints.get("max_candidate_evidence_nodes", 5)
            ),
            max_signed_edges_per_candidate_evidence=int(
                constraints.get("max_signed_edges_per_candidate_evidence", 2)
            ),
            exact_selected_edges=int(constraints.get("exact_selected_edges", 1)),
            strict_single_pointer_mode=bool(
                constraints.get("strict_single_pointer_mode", True)
            ),
        ),
    )


def validate_trace_rec_graph(
    payload: dict[str, Any] | TraceRecGraph,
    *,
    constraints: TraceRecV1Constraints | None = None,
) -> ValidationResult:
    """Validate a raw payload or parsed TRACE-Rec graph."""

    constraints = constraints or TraceRecV1Constraints()
    result = ValidationResult()

    if isinstance(payload, TraceRecGraph):
        graph = payload
    else:
        graph = _coerce_graph(payload, result)
        if graph is None:
            return result

    _validate_metadata(graph.metadata, result)
    _validate_nodes(graph, constraints, result)
    _validate_edges(graph, constraints, result)
    _validate_registry_grounding(graph, result)
    return result


def parse_trace_rec_graph(
    payload: dict[str, Any],
    *,
    constraints: TraceRecV1Constraints | None = None,
    strict: bool = True,
) -> TraceRecGraph:
    """Parse a raw payload into a typed TRACE-Rec graph object."""

    constraints = constraints or TraceRecV1Constraints()
    result = ValidationResult()
    graph = _coerce_graph(payload, result)
    if graph is None:
        raise TraceRecGraphValidationError(result)

    validation = validate_trace_rec_graph(graph, constraints=constraints)
    if strict and not validation.valid:
        raise TraceRecGraphValidationError(validation)
    return graph


def _coerce_graph(payload: dict[str, Any], result: ValidationResult) -> TraceRecGraph | None:
    """Convert a raw graph payload into dataclasses before semantic checks."""

    if not isinstance(payload, dict):
        result.add("invalid_graph", "graph payload must be a mapping", "graph")
        return None

    metadata_raw = payload.get("metadata")
    nodes_raw = payload.get("nodes")
    edges_raw = payload.get("edges")

    if not isinstance(metadata_raw, dict):
        result.add("missing_metadata", "metadata must be a mapping", "metadata")
        return None
    if not isinstance(nodes_raw, list):
        result.add("missing_nodes", "nodes must be a list", "nodes")
        return None
    if not isinstance(edges_raw, list):
        result.add("missing_edges", "edges must be a list", "edges")
        return None

    candidate_ids = metadata_raw.get("candidate_ids")
    if not isinstance(candidate_ids, list):
        candidate_ids = []

    metadata = GraphMetadata(
        task=str(metadata_raw.get("task", "")),
        user_id=str(metadata_raw.get("user_id", "")),
        candidate_ids=tuple(str(item) for item in candidate_ids),
        context=metadata_raw.get("context"),
    )

    nodes: list[TraceRecNode] = []
    for index, node_raw in enumerate(nodes_raw):
        node = _coerce_node(node_raw, result, f"nodes[{index}]")
        if node is not None:
            nodes.append(node)

    edges: list[GraphEdge] = []
    for index, edge_raw in enumerate(edges_raw):
        edge = _coerce_edge(edge_raw, result, f"edges[{index}]")
        if edge is not None:
            edges.append(edge)

    if not result.valid:
        return None

    return TraceRecGraph(metadata=metadata, nodes=tuple(nodes), edges=tuple(edges))


def _coerce_node(
    node_raw: Any, result: ValidationResult, path: str
) -> TraceRecNode | None:
    """Convert one raw node into a typed node dataclass."""

    if not isinstance(node_raw, dict):
        result.add("invalid_node", "node must be a mapping", path)
        return None

    node_type = node_raw.get("type")
    node_id = str(node_raw.get("id", ""))

    if node_type == GraphNodeType.PREFERENCE_STATE.value:
        return PreferenceStateNode(
            id=node_id,
            summary=str(node_raw.get("summary", "")),
            polarity=_safe_enum(
                PreferencePolarity, node_raw.get("polarity"), result, f"{path}.polarity"
            )
            or PreferencePolarity.POSITIVE,
            horizon=_safe_enum(
                PreferenceHorizon, node_raw.get("horizon"), result, f"{path}.horizon"
            )
            or PreferenceHorizon.RECENT,
            evidence_refs=tuple(str(item) for item in node_raw.get("evidence_refs", [])),
            source=_safe_enum(
                PreferenceSource, node_raw.get("source"), result, f"{path}.source"
            )
            or PreferenceSource.HISTORY,
        )

    if node_type == GraphNodeType.CANDIDATE_EVIDENCE.value:
        rank_prior_raw = node_raw.get("rank_prior")
        rank_prior = float(rank_prior_raw) if rank_prior_raw is not None else None
        retrieval_source = node_raw.get("retrieval_source")
        return CandidateEvidenceNode(
            id=node_id,
            candidate_id=str(node_raw.get("candidate_id", "")),
            feature_refs=tuple(str(item) for item in node_raw.get("feature_refs", [])),
            summary=str(node_raw.get("summary", "")),
            rank_prior=rank_prior,
            retrieval_source=None if retrieval_source is None else str(retrieval_source),
        )

    if node_type == GraphNodeType.DECISION.value:
        return DecisionNode(
            id=node_id,
            selected_item_id=str(node_raw.get("selected_item_id", "")),
        )

    result.add("invalid_node_type", f"unsupported node type {node_type!r}", f"{path}.type")
    return None


def _coerce_edge(edge_raw: Any, result: ValidationResult, path: str) -> GraphEdge | None:
    """Convert one raw edge into a typed edge dataclass."""

    if not isinstance(edge_raw, dict):
        result.add("invalid_edge", "edge must be a mapping", path)
        return None

    edge_type = _safe_enum(GraphEdgeType, edge_raw.get("type"), result, f"{path}.type")
    if edge_type is None:
        return None

    strength = edge_raw.get("strength")
    reason_span = edge_raw.get("reason_span")
    return GraphEdge(
        source=str(edge_raw.get("source", "")),
        target=str(edge_raw.get("target", "")),
        type=edge_type,
        strength=None if strength is None else str(strength),
        reason_span=None if reason_span is None else str(reason_span),
    )


def _safe_enum(
    enum_cls: type[Enum], value: Any, result: ValidationResult, path: str
) -> Enum | None:
    """Cast a raw string into an enum member and record validation failures."""

    try:
        return enum_cls(value)  # type: ignore[misc]
    except Exception:
        result.add("invalid_enum_value", f"invalid value {value!r}", path)
        return None


def _validate_metadata(metadata: GraphMetadata, result: ValidationResult) -> None:
    """Validate graph-level metadata."""

    if metadata.task != "next_item_recommendation":
        result.add(
            "invalid_task",
            "task must equal 'next_item_recommendation'",
            "metadata.task",
        )
    if not metadata.user_id:
        result.add("missing_user_id", "user_id is required", "metadata.user_id")
    if not metadata.candidate_ids:
        result.add(
            "missing_candidate_ids",
            "candidate_ids must contain at least one candidate",
            "metadata.candidate_ids",
        )
    elif len(set(metadata.candidate_ids)) != len(metadata.candidate_ids):
        result.add(
            "duplicate_candidate_ids",
            "candidate_ids must be unique",
            "metadata.candidate_ids",
        )
    if metadata.context is not None and not isinstance(metadata.context, dict):
        result.add(
            "invalid_context",
            "metadata.context must be a mapping when provided",
            "metadata.context",
        )


def _validate_nodes(
    graph: TraceRecGraph, constraints: TraceRecV1Constraints, result: ValidationResult
) -> None:
    """Validate node-level constraints."""

    ids = [node.id for node in graph.nodes]
    if len(ids) != len(set(ids)):
        result.add("duplicate_node_id", "node ids must be unique", "nodes")

    preference_nodes = [n for n in graph.nodes if isinstance(n, PreferenceStateNode)]
    candidate_nodes = [n for n in graph.nodes if isinstance(n, CandidateEvidenceNode)]
    decision_nodes = [n for n in graph.nodes if isinstance(n, DecisionNode)]

    if not preference_nodes:
        result.add(
            "missing_preference_state",
            "at least one preference_state node is required",
            "nodes",
        )
    if not candidate_nodes:
        result.add(
            "missing_candidate_evidence",
            "at least one candidate_evidence node is required",
            "nodes",
        )
    if len(decision_nodes) != 1:
        result.add(
            "invalid_decision_count",
            "exactly one decision node is required",
            "nodes",
        )

    if len(preference_nodes) > constraints.max_preference_state_nodes:
        result.add(
            "too_many_preference_states",
            f"at most {constraints.max_preference_state_nodes} preference_state nodes allowed",
            "nodes",
        )
    if len(candidate_nodes) > constraints.max_candidate_evidence_nodes:
        result.add(
            "too_many_candidate_evidence_nodes",
            f"at most {constraints.max_candidate_evidence_nodes} candidate_evidence nodes allowed",
            "nodes",
        )

    candidate_id_set = set(graph.metadata.candidate_ids)
    for node in preference_nodes:
        if not node.summary:
            result.add("missing_summary", "summary is required", f"nodes[{node.id}].summary")
        if constraints.require_evidence_grounding and not node.evidence_refs:
            result.add(
                "missing_evidence_refs",
                "preference_state must provide evidence_refs",
                f"nodes[{node.id}].evidence_refs",
            )
        if constraints.strict_single_pointer_mode and len(node.evidence_refs) != 1:
            result.add(
                "invalid_pointer_count",
                "strict mode requires exactly one evidence_ref",
                f"nodes[{node.id}].evidence_refs",
            )

    for node in candidate_nodes:
        if not node.summary:
            result.add("missing_summary", "summary is required", f"nodes[{node.id}].summary")
        if node.candidate_id not in candidate_id_set:
            result.add(
                "candidate_not_in_set",
                "candidate_id must belong to metadata.candidate_ids",
                f"nodes[{node.id}].candidate_id",
            )
        if constraints.require_evidence_grounding and not node.feature_refs:
            result.add(
                "missing_feature_refs",
                "candidate_evidence must provide feature_refs",
                f"nodes[{node.id}].feature_refs",
            )
        if constraints.strict_single_pointer_mode and len(node.feature_refs) != 1:
            result.add(
                "invalid_pointer_count",
                "strict mode requires exactly one feature_ref",
                f"nodes[{node.id}].feature_refs",
            )

    for node in decision_nodes:
        if node.selected_item_id not in candidate_id_set:
            result.add(
                "selected_item_not_in_set",
                "decision.selected_item_id must belong to metadata.candidate_ids",
                f"nodes[{node.id}].selected_item_id",
            )


def _validate_edges(
    graph: TraceRecGraph, constraints: TraceRecV1Constraints, result: ValidationResult
) -> None:
    """Validate edge-level constraints."""

    node_lookup = graph.node_by_id()
    signed_incoming_by_candidate: dict[str, int] = {}
    signed_relation_pairs: set[tuple[str, str, GraphEdgeType]] = set()
    selected_edges: list[GraphEdge] = []

    for index, edge in enumerate(graph.edges):
        path = f"edges[{index}]"
        source = node_lookup.get(edge.source)
        target = node_lookup.get(edge.target)

        if source is None:
            result.add("missing_edge_source", "source node id does not exist", f"{path}.source")
            continue
        if target is None:
            result.add("missing_edge_target", "target node id does not exist", f"{path}.target")
            continue

        if edge.type in {GraphEdgeType.SUPPORTS, GraphEdgeType.CONFLICTS}:
            if not isinstance(source, PreferenceStateNode) or not isinstance(
                target, CandidateEvidenceNode
            ):
                result.add(
                    "invalid_signed_edge_shape",
                    "signed edges must go from preference_state to candidate_evidence",
                    path,
                )
                continue
            signed_incoming_by_candidate[target.id] = (
                signed_incoming_by_candidate.get(target.id, 0) + 1
            )
            signed_relation_pairs.add((source.id, target.id, edge.type))

        elif edge.type == GraphEdgeType.SELECTED:
            selected_edges.append(edge)
            if not isinstance(source, CandidateEvidenceNode) or not isinstance(
                target, DecisionNode
            ):
                result.add(
                    "invalid_selected_edge_shape",
                    "selected edges must go from candidate_evidence to decision",
                    path,
                )

    for candidate_node_id, count in signed_incoming_by_candidate.items():
        if count > constraints.max_signed_edges_per_candidate_evidence:
            result.add(
                "too_many_signed_edges",
                (
                    "candidate_evidence exceeds max signed edges "
                    f"({constraints.max_signed_edges_per_candidate_evidence})"
                ),
                f"nodes[{candidate_node_id}]",
            )

    for source_id, target_id, _ in list(signed_relation_pairs):
        if (
            source_id,
            target_id,
            GraphEdgeType.SUPPORTS,
        ) in signed_relation_pairs and (
            source_id,
            target_id,
            GraphEdgeType.CONFLICTS,
        ) in signed_relation_pairs:
            result.add(
                "contradictory_signed_edges",
                "the same preference_state cannot both support and conflict with one candidate",
                f"edges[{source_id}->{target_id}]",
            )

    if len(selected_edges) != constraints.exact_selected_edges:
        result.add(
            "invalid_selected_edge_count",
            f"exactly {constraints.exact_selected_edges} selected edge(s) required",
            "edges",
        )
    else:
        selected_edge = selected_edges[0]
        decision_nodes = [node for node in graph.nodes if isinstance(node, DecisionNode)]
        if decision_nodes:
            decision = decision_nodes[0]
            source = node_lookup.get(selected_edge.source)
            if isinstance(source, CandidateEvidenceNode):
                if source.candidate_id != decision.selected_item_id:
                    result.add(
                        "selected_edge_mismatch",
                        "selected edge source candidate_id must match decision.selected_item_id",
                        "edges",
                    )


def _validate_registry_grounding(graph: TraceRecGraph, result: ValidationResult) -> None:
    """Validate evidence and feature refs against metadata-provided registries when present."""

    evidence_registry = graph.metadata.evidence_registry()
    candidate_feature_registries = {
        node.id: graph.metadata.feature_registry(node.candidate_id)
        for node in graph.nodes
        if isinstance(node, CandidateEvidenceNode)
    }

    for node in graph.nodes:
        if isinstance(node, PreferenceStateNode) and evidence_registry:
            for ref in node.evidence_refs:
                if ref not in evidence_registry:
                    result.add(
                        "unknown_evidence_ref",
                        "evidence_ref must belong to metadata.context registry",
                        f"nodes[{node.id}].evidence_refs",
                    )
        if isinstance(node, CandidateEvidenceNode):
            feature_registry = candidate_feature_registries.get(node.id, set())
            if not feature_registry:
                continue
            for ref in node.feature_refs:
                if ref not in feature_registry:
                    result.add(
                        "unknown_feature_ref",
                        "feature_ref must belong to the candidate feature registry",
                        f"nodes[{node.id}].feature_refs",
                    )
