"""Strict, provider-neutral contracts shared by the AGI workspace."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Self
from uuid import UUID, uuid4

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    """Return an aware UTC timestamp suitable for persisted records."""

    return datetime.now(timezone.utc)


Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$",
    ),
]
ModelId = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)
]
Description = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)
]
ShortText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=16_384)
]
LongText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000_000)
]
ContentText = Annotated[
    str, StringConstraints(strip_whitespace=False, max_length=2_000_000)
]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-fA-F]{64}$")]
JsonObject = dict[str, JsonValue]


class ContractModel(BaseModel):
    """Common configuration for all persisted and transported contracts."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
    )


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class StepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationVerdict(StrEnum):
    PASSED = "passed"
    PARTIAL = "partial"
    FAILED = "failed"


class AnswerSource(StrEnum):
    """Who authored the final answer text, so host fallbacks never count as model skill."""

    MODEL = "model"
    MODEL_CORRECTION = "model_correction"
    HOST_FALLBACK = "host_fallback"


class ErrorDetail(ContractModel):
    code: Identifier
    message: ShortText
    component: Identifier | None = None
    retryable: bool = False
    details: JsonObject = Field(default_factory=dict)
    occurred_at: AwareDatetime = Field(default_factory=utc_now)


class EvidenceReference(ContractModel):
    evidence_id: UUID = Field(default_factory=uuid4)
    source_id: Identifier
    source_path: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32_768)
    ]
    locator: (
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
        ]
        | None
    ) = None
    excerpt: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=False, min_length=1, max_length=65_536),
        ]
        | None
    ) = None
    content_sha256: Sha256Digest | None = None
    metadata: JsonObject = Field(default_factory=dict)


class EvidenceItem(ContractModel):
    evidence_id: UUID = Field(default_factory=uuid4)
    source_id: Identifier
    source_hash: Sha256Digest
    source_type: Identifier = "document"
    excerpt: Annotated[
        str, StringConstraints(strip_whitespace=False, min_length=1, max_length=4096)
    ]
    retrieval_query: ShortText
    retrieval_tool: Identifier
    retrieved_at: AwareDatetime = Field(default_factory=utc_now)
    metadata: JsonObject = Field(default_factory=dict)


class EvidenceBundle(ContractModel):
    bundle_id: UUID = Field(default_factory=uuid4)
    items: list[EvidenceItem] = Field(default_factory=list, max_length=512)
    source_count: int = Field(ge=0, le=512)
    evidence_count: int = Field(ge=0, le=512)
    bundle_hash: Sha256Digest

    @model_validator(mode="after")
    def counts_and_ids_must_match(self) -> Self:
        if self.evidence_count != len(self.items):
            raise ValueError("evidence_count must equal items length")
        if self.source_count != len({item.source_id for item in self.items}):
            raise ValueError("source_count must equal distinct source count")
        if len({item.evidence_id for item in self.items}) != len(self.items):
            raise ValueError("evidence IDs must be unique")
        return self


class Citation(ContractModel):
    citation_id: UUID = Field(default_factory=uuid4)
    evidence_id: UUID
    claim: ShortText | None = None
    label: (
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)
        ]
        | None
    ) = None


class ModelMessage(ContractModel):
    role: MessageRole
    content: LongText
    name: Identifier | None = None
    tool_call_id: UUID | None = None


class ModelRequest(ContractModel):
    request_id: UUID = Field(default_factory=uuid4)
    model: ModelId
    messages: Annotated[list[ModelMessage], Field(min_length=1, max_length=256)]
    response_schema: JsonObject | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=131_072)
    seed: int | None = Field(default=None, ge=0, le=9_223_372_036_854_775_807)
    metadata: JsonObject = Field(default_factory=dict)


class ModelUsage(ContractModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0.0)


class ModelResponse(ContractModel):
    request_id: UUID
    model: ModelId
    content: ContentText = ""
    structured_output: JsonObject | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)
    raw_metadata: JsonObject = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def require_content_or_structure(self) -> Self:
        if not self.content.strip() and self.structured_output is None:
            raise ValueError("content or structured_output must be provided")
        return self


class ApprovalPolicy(ContractModel):
    default_requires_human_approval: bool = False
    tools_requiring_approval: list[Identifier] = Field(
        default_factory=list, max_length=128
    )
    risk_levels_requiring_approval: list[RiskLevel] = Field(
        default_factory=lambda: [RiskLevel.HIGH, RiskLevel.CRITICAL], max_length=4
    )

    @field_validator("tools_requiring_approval", "risk_levels_requiring_approval")
    @classmethod
    def values_must_be_unique(cls, value: list[object]) -> list[object]:
        if len(value) != len(set(value)):
            raise ValueError("values must be unique")
        return value


class AgentProfile(ContractModel):
    id: Identifier
    description: Description
    model: ModelId
    allowed_tools: list[Identifier] = Field(default_factory=list, max_length=128)
    permissions: list[Identifier] = Field(default_factory=list, max_length=256)
    allowed_document_roots: list[
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=32_768),
        ]
    ] = Field(default_factory=list, max_length=32)
    max_steps: int = Field(default=12, ge=1, le=100)
    max_tool_calls: int = Field(default=8, ge=0, le=100)
    approval_policy: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("allowed_tools", "permissions", "allowed_document_roots")
    @classmethod
    def values_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("values must be unique")
        return value

    @field_validator("allowed_document_roots")
    @classmethod
    def document_roots_must_be_local_and_absolute(cls, roots: list[str]) -> list[str]:
        for root in roots:
            if root.startswith(("\\\\", "//")):
                raise ValueError("UNC and device document roots are not allowed")
            if not (
                PureWindowsPath(root).is_absolute() or PurePosixPath(root).is_absolute()
            ):
                raise ValueError("document roots must be absolute")
        return roots


class TaskStep(ContractModel):
    step_id: UUID = Field(default_factory=uuid4)
    description: Description
    status: StepStatus = StepStatus.PENDING
    tool_name: Identifier | None = None
    tool_arguments: JsonObject = Field(default_factory=dict)
    success_criteria: ShortText | None = None


class TaskPlan(ContractModel):
    plan_id: UUID = Field(default_factory=uuid4)
    goal: LongText
    steps: Annotated[list[TaskStep], Field(min_length=1, max_length=100)]
    created_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def step_ids_must_be_unique(self) -> Self:
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step_id values must be unique")
        return self


class TaskRequirements(ContractModel):
    """Immutable host-owned requirements for one task execution."""

    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        str_strip_whitespace=True,
        validate_default=True,
        frozen=True,
    )

    required_capabilities: list[Identifier] = Field(default_factory=list, max_length=128)
    requires_grounding: bool = False
    requires_citations: bool = False
    minimum_citation_count: int = Field(default=0, ge=0, le=512)
    requires_uncertainty_statement: bool = False
    allowed_capabilities: list[Identifier] | None = Field(default=None, max_length=128)
    forbidden_capabilities: list[Identifier] = Field(default_factory=list, max_length=128)
    maximum_tool_calls: int | None = Field(default=None, ge=0, le=100)
    requirement_policy_version: Identifier = "task-requirements.v1"
    task_subjects: list[ShortText] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_capability_policy(self) -> Self:
        collections = (
            self.required_capabilities,
            self.allowed_capabilities or [],
            self.forbidden_capabilities,
        )
        if any(len(values) != len(set(values)) for values in collections):
            raise ValueError("capability lists must contain unique names")
        conflict = set(self.required_capabilities) & set(self.forbidden_capabilities)
        if conflict:
            raise ValueError(
                "capabilities cannot be both required and forbidden: "
                + ", ".join(sorted(conflict))
            )
        if self.requires_citations and self.minimum_citation_count < 1:
            raise ValueError(
                "requires_citations=true requires minimum_citation_count >= 1"
            )
        return self


class PlanRequirementResult(ContractModel):
    valid: bool
    required_capabilities: list[Identifier] = Field(default_factory=list, max_length=128)
    proposed_capabilities: list[Identifier] = Field(default_factory=list, max_length=128)
    missing_capabilities: list[Identifier] = Field(default_factory=list, max_length=128)
    forbidden_capabilities_proposed: list[Identifier] = Field(
        default_factory=list, max_length=128
    )
    unknown_capabilities_proposed: list[Identifier] = Field(
        default_factory=list, max_length=128
    )
    correction_attempted: bool = False
    correction_succeeded: bool = False
    validation_messages: list[ShortText] = Field(default_factory=list, max_length=256)


class ToolArgumentValidationResult(ContractModel):
    tool_name: Identifier
    valid: bool
    schema_errors: list[ShortText] = Field(default_factory=list, max_length=128)
    semantic_errors: list[ShortText] = Field(default_factory=list, max_length=128)
    unknown_arguments: list[Identifier] = Field(default_factory=list, max_length=128)
    missing_arguments: list[Identifier] = Field(default_factory=list, max_length=128)
    repair_attempted: bool = False
    repair_succeeded: bool = False
    final_valid: bool
    dispatch_allowed: bool
    validation_messages: list[ShortText] = Field(default_factory=list, max_length=256)


class ToolArgumentRepairHistory(ContractModel):
    tool_name: Identifier
    initial_arguments_redacted: JsonObject = Field(default_factory=dict)
    validation_result: ToolArgumentValidationResult
    repaired_arguments_redacted: JsonObject | None = None
    repaired_validation_result: ToolArgumentValidationResult | None = None
    attempt_count: int = Field(ge=1, le=2)


class ResponseRequirementResult(ContractModel):
    valid: bool
    citations_required: bool
    citation_count: int = Field(ge=0, le=512)
    citation_requirement_satisfied: bool
    grounding_required: bool
    grounding_evidence_present: bool
    uncertainty_required: bool
    uncertainty_statement_present: bool
    missing_requirements: list[Identifier] = Field(default_factory=list, max_length=128)
    correction_attempted: bool = False
    correction_succeeded: bool = False
    validation_messages: list[ShortText] = Field(default_factory=list, max_length=256)


class ToolDefinition(ContractModel):
    name: Identifier
    description: Description
    input_schema: JsonObject
    output_schema: JsonObject
    risk_level: RiskLevel
    permissions_required: list[Identifier] = Field(default_factory=list, max_length=64)
    human_approval_required: bool = False
    timeout_seconds: float = Field(default=10.0, gt=0.0, le=300.0)

    @field_validator("permissions_required")
    @classmethod
    def permissions_must_be_unique(cls, permissions: list[str]) -> list[str]:
        if len(permissions) != len(set(permissions)):
            raise ValueError("permissions_required must be unique")
        return permissions


class ToolRequest(ContractModel):
    request_id: UUID = Field(default_factory=uuid4)
    tool_name: Identifier
    arguments: JsonObject = Field(default_factory=dict)
    step_id: UUID | None = None


class ToolResult(ContractModel):
    request_id: UUID
    tool_name: Identifier
    success: bool
    output: JsonObject = Field(default_factory=dict)
    error: ShortText | None = None
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=256)
    started_at: AwareDatetime
    completed_at: AwareDatetime
    duration_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def completion_must_follow_start(self) -> Self:
        if self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        return self


class PermissionDecision(ContractModel):
    request_id: UUID
    tool_name: Identifier
    allowed: bool
    reason: ShortText
    permissions_checked: list[Identifier] = Field(default_factory=list, max_length=256)
    human_approval_required: bool = False
    decided_at: AwareDatetime = Field(default_factory=utc_now)


class RuleBasedCheck(ContractModel):
    name: Identifier
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    critical: bool = False
    details: ShortText | None = None
    evidence_ids: list[UUID] = Field(default_factory=list, max_length=256)


class EvaluationScores(ContractModel):
    task_completion: float = Field(ge=0.0, le=1.0)
    factual_grounding: float = Field(ge=0.0, le=1.0)
    plan_adherence: float = Field(ge=0.0, le=1.0)
    tool_correctness: float = Field(ge=0.0, le=1.0)
    unsupported_claims: float = Field(
        ge=0.0, le=1.0, description="1.0 means no unsupported claims were detected."
    )
    output_usefulness: float = Field(ge=0.0, le=1.0)


class ModelCritique(ContractModel):
    model_id: ModelId
    prompt_version: Identifier
    critique: LongText | None = None
    suggestions: list[ShortText] = Field(default_factory=list, max_length=128)
    error: ShortText | None = None
    evaluated_at: AwareDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def require_critique_or_error(self) -> Self:
        if self.critique is None and self.error is None:
            raise ValueError("critique or error must be provided")
        return self


class EvaluationResult(ContractModel):
    evaluation_id: UUID = Field(default_factory=uuid4)
    run_id: UUID | None = None
    scores: EvaluationScores
    rule_based_checks: list[RuleBasedCheck] = Field(
        default_factory=list, max_length=256
    )
    model_critique: ModelCritique | None = None
    unsupported_claims: list[ShortText] = Field(default_factory=list, max_length=256)
    verdict: EvaluationVerdict
    evaluated_at: AwareDatetime = Field(default_factory=utc_now)


class AgentRequest(ContractModel):
    request_id: UUID = Field(default_factory=uuid4)
    goal: LongText
    agent_profile_id: Identifier = "researcher"
    context: JsonObject = Field(default_factory=dict)
    metadata: JsonObject = Field(default_factory=dict)
    created_at: AwareDatetime = Field(default_factory=utc_now)


class AgentResponse(ContractModel):
    request_id: UUID
    run_id: UUID
    status: RunStatus
    answer: LongText
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=512)
    citations: list[Citation] = Field(default_factory=list, max_length=512)
    evaluation: EvaluationResult | None = None
    errors: list[ErrorDetail] = Field(default_factory=list, max_length=256)
    completed_at: AwareDatetime = Field(default_factory=utc_now)


class RunRecord(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: UUID = Field(default_factory=uuid4)
    request: AgentRequest
    profile: AgentProfile
    profile_hash: Sha256Digest | None = None
    model_id: ModelId
    model_parameters: JsonObject = Field(default_factory=dict)
    plan: TaskPlan | None = None
    task_requirements: TaskRequirements | None = None
    plan_requirement_result: PlanRequirementResult | None = None
    plan_requirement_results: list[PlanRequirementResult] = Field(
        default_factory=list, max_length=2
    )
    response_requirement_result: ResponseRequirementResult | None = None
    response_requirement_results: list[ResponseRequirementResult] = Field(
        default_factory=list, max_length=2
    )
    tool_argument_histories: list[ToolArgumentRepairHistory] = Field(
        default_factory=list, max_length=512
    )
    evidence_bundle: EvidenceBundle | None = None
    permission_decisions: list[PermissionDecision] = Field(
        default_factory=list, max_length=512
    )
    tool_requests: list[ToolRequest] = Field(default_factory=list, max_length=512)
    tool_results: list[ToolResult] = Field(default_factory=list, max_length=512)
    response: AgentResponse | None = None
    answer_source: AnswerSource | None = None
    evaluation: EvaluationResult | None = None
    status: RunStatus = RunStatus.PENDING
    started_at: AwareDatetime = Field(default_factory=utc_now)
    completed_at: AwareDatetime | None = None
    duration_ms: float | None = Field(default=None, ge=0.0)
    errors: list[ErrorDetail] = Field(default_factory=list, max_length=256)
    usage: ModelUsage = Field(default_factory=ModelUsage)
    input_hash: Sha256Digest | None = None
    config_hash: Sha256Digest | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_record_links_and_timing(self) -> Self:
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("completed_at must not precede started_at")
        if self.response is not None:
            if self.response.request_id != self.request.request_id:
                raise ValueError("response.request_id must match request.request_id")
            if self.response.run_id != self.run_id:
                raise ValueError("response.run_id must match run_id")
        return self


__all__ = [
    "AgentProfile",
    "AgentRequest",
    "AgentResponse",
    "ApprovalPolicy",
    "Citation",
    "ContractModel",
    "ErrorDetail",
    "EvaluationResult",
    "EvaluationScores",
    "EvaluationVerdict",
    "EvidenceReference",
    "EvidenceBundle",
    "EvidenceItem",
    "MessageRole",
    "ModelCritique",
    "ModelMessage",
    "ModelRequest",
    "ModelResponse",
    "ModelUsage",
    "PermissionDecision",
    "RiskLevel",
    "RuleBasedCheck",
    "PlanRequirementResult",
    "ResponseRequirementResult",
    "RunRecord",
    "RunStatus",
    "StepStatus",
    "TaskPlan",
    "TaskRequirements",
    "TaskStep",
    "ToolDefinition",
    "ToolArgumentRepairHistory",
    "ToolArgumentValidationResult",
    "ToolRequest",
    "ToolResult",
    "utc_now",
]
