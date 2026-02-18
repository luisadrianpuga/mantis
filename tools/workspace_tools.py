from workspace import WorkspaceIndex


workspace_index = WorkspaceIndex()


def tree(_: str = "") -> str:
    return workspace_index.tree()


def read_file(path: str) -> str:
    path = path.strip()
    if not path:
        return "workspace.read_file requires a file path."
    try:
        return workspace_index.read_file(path)
    except ValueError as exc:
        return f"Workspace read error: {exc}"


def search(query: str) -> str:
    return workspace_index.search(query)
