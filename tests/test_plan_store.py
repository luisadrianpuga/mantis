from agent.plan_store import PlanStore


def test_plan_store_lifecycle(tmp_path):
    store = PlanStore(tmp_path / "plans")
    created = store.create_plan("goal", ["task1", "task2"])

    assert created["status"] == "running"

    updated = store.mark_task_completed(created["id"], 0)
    assert updated["completed_tasks"] == [0]

    latest = store.latest_running()
    assert latest is not None
    assert latest["id"] == created["id"]

    done = store.set_status(created["id"], "completed")
    assert done["status"] == "completed"
    assert store.latest_running() is None
