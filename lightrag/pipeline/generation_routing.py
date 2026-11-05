from __future__ import annotations

import os

from .coordinator import (
    PipelineCoordinator,
    WorkspaceGenerationStatus,
    create_coordinator_from_env,
)


def generation_routing_enabled() -> bool:
    return os.getenv("PIPELINE_GENERATION_ROUTING_ENABLED", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


async def resolve_storage_workspace_id(
    workspace_id: str,
    coordinator: PipelineCoordinator | None = None,
) -> str:
    """Resolve the active physical storage workspace for a logical workspace."""
    logical_workspace_id = workspace_id or "default"
    if not generation_routing_enabled():
        return logical_workspace_id

    owns_coordinator = coordinator is None
    active_coordinator = coordinator or create_coordinator_from_env()
    try:
        generation_id = await active_coordinator.get_active_workspace_generation(
            logical_workspace_id
        )
        if generation_id is None:
            return logical_workspace_id

        generation = await active_coordinator.get_workspace_generation(
            logical_workspace_id, generation_id
        )
        if generation is None:
            raise RuntimeError(
                "Active workspace generation pointer references a missing "
                f"generation: workspace_id={logical_workspace_id}, "
                f"generation_id={generation_id}"
            )
        if generation.status != WorkspaceGenerationStatus.ACTIVE:
            raise RuntimeError(
                "Active workspace generation pointer references a generation "
                f"with status={generation.status.value}: "
                f"workspace_id={logical_workspace_id}, generation_id={generation_id}"
            )
        return generation.storage_workspace_id
    finally:
        if owns_coordinator:
            await active_coordinator.finalize()
