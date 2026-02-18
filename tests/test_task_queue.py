from worker.task_queue import TaskQueue


def test_task_queue_persistence(tmp_path):
    queue = TaskQueue(tmp_path / "queue.json")
    item = queue.enqueue("Add a health endpoint")
    assert item["retries"] == 0

    fetched = queue.next_pending()
    assert fetched is not None
    assert fetched["id"] == item["id"]
    assert fetched["status"] == "running"

    updated = queue.set_status(item["id"], "done")
    assert updated is not None
    assert updated["status"] == "done"

    queue_reloaded = TaskQueue(tmp_path / "queue.json")
    items = queue_reloaded.list_items()
    assert len(items) == 1
    assert items[0]["status"] == "done"


def test_task_queue_has_goal_case_insensitive(tmp_path):
    queue = TaskQueue(tmp_path / "queue.json")
    queue.enqueue("Add a health endpoint")

    assert queue.has_goal("add a health endpoint")
    assert queue.has_goal("  ADD A HEALTH ENDPOINT  ")
    assert queue.has_goal("add a health")
    assert queue.has_goal("add a health endpoint and tests")
    assert not queue.has_goal("Different goal")


def test_task_queue_increment_retries(tmp_path):
    queue = TaskQueue(tmp_path / "queue.json")
    item = queue.enqueue("Task that may fail")

    queue.increment_retries(item["id"])
    queue.increment_retries(item["id"])

    items = queue.list_items()
    assert items[0]["retries"] == 2
