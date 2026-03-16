import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _import_router_module(module_path: str):
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        module = importlib.import_module(module_path)
        return importlib.reload(module)
    finally:
        sys.argv = original_argv


@pytest.mark.offline
def test_query_endpoint_uses_workspace_id_from_body():
    query_routes = _import_router_module("lightrag.api.routers.query_routes")

    default_rag = SimpleNamespace(workspace="default")
    workspace_calls: list[str] = []

    rag_ws = SimpleNamespace()
    rag_ws.aquery_llm = AsyncMock(
        return_value={
            "llm_response": {"content": "ok"},
            "data": {"references": []},
        }
    )

    async def get_rag_for_workspace(workspace_id: str):
        workspace_calls.append(workspace_id)
        return rag_ws

    app = FastAPI()
    app.include_router(
        query_routes.create_query_routes(
            default_rag,
            api_key=None,
            top_k=10,
            get_rag_for_workspace=get_rag_for_workspace,
        )
    )

    client = TestClient(app)
    resp = client.post(
        "/query",
        json={
            "query": "what is ai",
            "workspace_id": "ws-body",
            "include_references": False,
        },
    )

    assert resp.status_code == 200
    assert workspace_calls == ["ws-body"]
    rag_ws.aquery_llm.assert_awaited_once()


@pytest.mark.offline
def test_query_data_endpoint_uses_workspace_id_from_header_when_body_missing():
    query_routes = _import_router_module("lightrag.api.routers.query_routes")

    default_rag = SimpleNamespace(workspace="default")
    workspace_calls: list[str] = []

    rag_ws = SimpleNamespace()
    rag_ws.aquery_data = AsyncMock(
        return_value={
            "status": "success",
            "message": "ok",
            "data": {"entities": [], "relationships": [], "chunks": [], "references": []},
            "metadata": {},
        }
    )

    async def get_rag_for_workspace(workspace_id: str):
        workspace_calls.append(workspace_id)
        return rag_ws

    app = FastAPI()
    app.include_router(
        query_routes.create_query_routes(
            default_rag,
            api_key=None,
            top_k=10,
            get_rag_for_workspace=get_rag_for_workspace,
        )
    )

    client = TestClient(app)
    resp = client.post(
        "/query/data",
        headers={"LIGHTRAG-WORKSPACE-ID": "ws-header"},
        json={"query": "what is ai"},
    )

    assert resp.status_code == 200
    assert workspace_calls == ["ws-header"]
    rag_ws.aquery_data.assert_awaited_once()


@pytest.mark.offline
def test_graph_get_labels_uses_workspace_id_query_param():
    graph_routes = _import_router_module("lightrag.api.routers.graph_routes")

    default_rag = SimpleNamespace(workspace="default")
    workspace_calls: list[str] = []

    rag_ws = SimpleNamespace()
    rag_ws.get_graph_labels = AsyncMock(return_value=["A", "B"])

    async def get_rag_for_workspace(workspace_id: str):
        workspace_calls.append(workspace_id)
        return rag_ws

    app = FastAPI()
    app.include_router(
        graph_routes.create_graph_routes(
            default_rag,
            api_key=None,
            get_rag_for_workspace=get_rag_for_workspace,
        )
    )

    client = TestClient(app)
    resp = client.get("/graph/label/list", params={"workspace_id": "ws-query"})

    assert resp.status_code == 200
    assert resp.json() == ["A", "B"]
    assert workspace_calls == ["ws-query"]
    rag_ws.get_graph_labels.assert_awaited_once()


@pytest.mark.offline
def test_graph_entity_edit_prefers_body_workspace_over_header():
    graph_routes = _import_router_module("lightrag.api.routers.graph_routes")

    default_rag = SimpleNamespace(workspace="default")
    workspace_calls: list[str] = []

    rag_ws = SimpleNamespace()
    rag_ws.aedit_entity = AsyncMock(
        return_value={"entity_name": "Tesla", "description": "updated"}
    )

    async def get_rag_for_workspace(workspace_id: str):
        workspace_calls.append(workspace_id)
        return rag_ws

    app = FastAPI()
    app.include_router(
        graph_routes.create_graph_routes(
            default_rag,
            api_key=None,
            get_rag_for_workspace=get_rag_for_workspace,
        )
    )

    client = TestClient(app)
    resp = client.post(
        "/graph/entity/edit",
        headers={"LIGHTRAG-WORKSPACE-ID": "ws-header"},
        json={
            "entity_name": "Tesla",
            "updated_data": {"description": "updated"},
            "workspace_id": "ws-body",
        },
    )

    assert resp.status_code == 200
    assert workspace_calls == ["ws-body"]
    rag_ws.aedit_entity.assert_awaited_once()
