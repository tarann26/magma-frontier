"""Two-level tool taxonomy.

Toolathlon tool ids are namespaced as "<namespace>-<tool>". Namespaces observed in
the corpus are grouped into coarse categories so n-gram features can be computed at
a level where vocabulary differences between tenants cannot carry the signal.
"""

_CATEGORY_BY_NAMESPACE = {
    "filesystem": "storage",
    "memory": "storage",
    "snowflake": "storage",
    "excel": "storage",
    "pdf": "storage",
    "terminal": "exec",
    "local": "exec",
    "github": "devtools",
    "huggingface": "devtools",
    "wandb": "devtools",
    "arxiv_local": "research",
    "scholarly": "research",
    "canvas": "research",
    "google_sheet": "productivity",
    "google_forms": "productivity",
    "notion": "productivity",
    "emails": "productivity",
    "google": "web",
    "google_map": "web",
    "fetch": "web",
    "yahoo": "web",
    "playwright_with_chunk": "web",
    "rail_12306": "commerce",
    "woocommerce": "commerce",
}

_MAX_DEPTH = 2


def namespace_of(tool_id: str) -> str:
    """Return the namespace prefix, or the whole id when there is no hyphen."""
    return tool_id.split("-", 1)[0]


def category_of(tool_id: str) -> str:
    """Return the coarse category for a tool id, or "other" if unmapped."""
    return _CATEGORY_BY_NAMESPACE.get(namespace_of(tool_id), "other")


def taxonomy_path(tool_id: str, depth: int) -> str:
    """Return the tool id generalized to `depth`: 0 category, 1 namespace, 2 full id."""
    if depth == 0:
        return category_of(tool_id)
    if depth == 1:
        return namespace_of(tool_id)
    if depth == _MAX_DEPTH:
        return tool_id
    raise ValueError(f"depth must be 0, 1 or {_MAX_DEPTH}, got {depth}")
