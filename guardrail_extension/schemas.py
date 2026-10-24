import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: str = Field(min_length=1, max_length=100)
    stage: Literal["before_model", "after_model"]
    payload: dict[str, Any]
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def text_payload(cls, value):
        # Other fields are ignored, never serialized into the text or trusted as policy.
        text = value.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 16000:
            raise ValueError("payload.text must be nonempty text of at most 16000 characters")
        return value


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["custom", "default"] = "custom"
    instructions: str | None = Field(default=None, min_length=1, max_length=8000)

    @model_validator(mode="after")
    def valid_mode(self):
        if self.mode == "custom" and (not self.instructions or not self.instructions.strip()):
            raise ValueError("custom mode requires instructions")
        if self.mode == "default" and self.instructions is not None:
            raise ValueError("default mode cannot override model policy")
        return self


class RegexRule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    deny_patterns: tuple[str, ...] = Field(min_length=1, max_length=16)

    @field_validator("deny_patterns")
    @classmethod
    def valid_patterns(cls, patterns):
        for pattern in patterns:
            if not pattern or len(pattern) > 500:
                raise ValueError("regex pattern must contain 1 to 500 characters")
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error:
                raise ValueError("invalid regex pattern") from None
        return patterns


class CheckSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    rail: str
    config_id: str
    policy_id: str
