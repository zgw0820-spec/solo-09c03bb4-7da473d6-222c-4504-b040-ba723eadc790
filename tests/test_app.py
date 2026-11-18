from __future__ import annotations

import hashlib
import io
import os
import threading
from pathlib import Path
from urllib.parse import unquote

import pytest
from starlette.testclient import TestClient

from app.main import create_app
from app.maintenance import run_inspection
from app.storage import ObjectStorage


@pytest.fixture()
def app_env(tmp_path: Path):
    app = create_app(tmp_path / "data", start_background=False)
    client = TestClient(app)
    with client:
        yield client, app.state.storage


def upload(client: TestClient, data: bytes, filename: str = "payload.bin"):
    return client.post(
        "/upload",
        files={"file": (filename, io.BytesIO(data), "application/octet-stream")},
        follow_redirects=False,
    )


def replica_path(storage: ObjectStorage, slot: int, version_id: str | None = None) -> Path:
    if version_id is None:
        version = storage.current_version()
        assert version is not None
        version_id = version.id
    row = storage.replica_rows(version_id)[slot]
    return storage.data_dir / row["relative_path"]


def inspect(client: TestClient):
    return client.post("/inspect", data={}, follow_redirects=False)


def test_upload_and_download_roundtrip(app_env):
    client, storage = app_env
    payload = os.urandom(4096)
    response = upload(client, payload, "报告 v1.bin")
    assert response.status_code == 303

    version = storage.current_version()
    assert version is not None
    assert version.digest == hashlib.sha256(payload).hexdigest()
    assert version.size == len(payload)

    for slot in (1, 2, 3):
        path = replica_path(storage, slot)
        assert path.exists()
        assert path.read_bytes() == payload

    download = client.get("/download")
    assert download.status_code == 200
    assert download.content == payload
    assert download.headers["x-object-sha256"] == version.digest
    assert download.headers["x-object-version"] == "1"
    assert "filename*=UTF-8''" in download.headers["content-disposition"]


def test_upload_size_limit(app_env):
    client, storage = app_env
    too_big = b"a" * (1024 * 1024 + 1)
    response = upload(client, too_big)
    assert response.status_code == 303
    assert "notice" in response.headers["location"]
    assert storage.current_version() is None

    exact = b"b" * (1024 * 1024)
    assert upload(client, exact).status_code == 303
    version = storage.current_version()
    assert version is not None and version.size == len(exact)
    assert client.get("/download").content == exact


def test_download_refuses_when_all_replicas_corrupt(app_env):
    client, storage = app_env
    payload = b"important-bytes" * 100
    upload(client, payload)

    for slot in (1, 2, 3):
        path = replica_path(storage, slot)
        path.write_bytes(b"broken" * 10)

    response = client.get("/download")
    assert response.status_code == 409
    assert "拒绝" in response.text
    assert client.get("/").status_code == 200
    assert "不可恢复" in client.get("/").text


def test_stale_replica_bytes_never_delivered(app_env):
    client, storage = app_env
    upload(client, b"first-version")
    first = storage.current_version()

    # 只破坏两份，仍有一份旧副本；随后替换为新版本并破坏新版本的一份副本，
    # 确保下载不会误用旧版本的遗留字节。
    replica_path(storage, 1).write_bytes(b"tampered")
    replica_path(storage, 3).write_bytes(b"tampered")
    assert client.get("/download").content == b"first-version"

    upload(client, b"second-version")
    second = storage.current_version()
    assert second.id != first.id
    replica_path(storage, 2).write_bytes(b"old-leftover")
    assert client.get("/download").content == b"second-version"

    for slot in (1, 2, 3):
        replica_path(storage, slot).write_bytes(b"tampered")
    assert client.get("/download").status_code == 409


def test_repair_from_verified_replica(app_env):
    client, storage = app_env
    payload = os.urandom(2048)
    upload(client, payload)
    version = storage.current_version()

    replica_path(storage, 2).write_bytes(b"x" * 100)
    assert client.get("/download").content == payload
    status = client.get("/api/status").json()
    assert status["readable"] == 2

    inspect(client)
    status = client.get("/api/status").json()
    assert status["readable"] == 3
    assert status["latest_inspection"]["state"] == "completed"
    assert replica_path(storage, 2).read_bytes() == payload
    assert client.get("/download").content == payload

    inspection = storage.recent_inspections(1)[0]
    assert inspection.version_id == version.id
    assert inspection.repaired_count == 1


def test_repair_unrecoverable_when_three_bad(app_env):
    client, storage = app_env
    upload(client, b"payload")
    for slot in (1, 2, 3):
        replica_path(storage, slot).write_bytes(b"broken")
    inspect(client)
    status = client.get("/api/status").json()
    assert status["latest_inspection"]["state"] == "unrecoverable"
    assert status["unrecoverable"] is True
    assert client.get("/download").status_code == 409
    assert "不可恢复" in client.get("/").text


def test_partial_repair_can_resume(app_env):
    client, storage = app_env
    payload = os.urandom(700)
    upload(client, payload)

    # 破坏两份，再只修复其中一份：模拟一次未完成的巡检修复。
    replica_path(storage, 2).write_bytes(b"bad")
    replica_path(storage, 3).write_bytes(b"bad")

    version = storage.current_version()
    inspection = storage.begin_inspection(version.id)
    storage.set_inspection_progress(inspection.id, "模拟中断前只修一份……")
    staged, digest = storage.stage_repair(version, 1, 2)
    assert storage.commit_repair(version.id, 2, staged, 1, digest) == "committed"

    checks = storage.check_replicas(version)
    assert [c.state for c in checks].count("verified") == 2

    # 中断的巡检在重启后标记为 interrupted，状态按磁盘实际副本显示。
    restarted = ObjectStorage(storage.data_dir)
    restarted.init()
    assert restarted.recent_inspections(1)[0].state == "interrupted"

    app = create_app(storage.data_dir, start_background=False)
    client = TestClient(app)
    with client:
        page = client.get("/")
        assert page.status_code == 200
        assert "2/3" in page.text
        assert client.get("/download").content == payload
        client.post("/inspect", data={}, follow_redirects=False)
        assert client.get("/api/status").json()["readable"] == 3


def test_interrupted_repair_recovers_after_restart(tmp_path: Path):
    data_dir = tmp_path / "data"
    payload = os.urandom(3333)

    storage = ObjectStorage(data_dir)
    storage.init()
    version = storage.create_version(payload, "restart.bin")

    for slot in (2, 3):
        path = data_dir / storage.replica_rows(version.id)[slot]["relative_path"]
        path.write_bytes(b"broken")

    # 模拟修复只完成一部分后进程崩溃：暂存文件残留 + 巡检停在 running。
    inspection = storage.begin_inspection(version.id)
    staged, digest = storage.stage_repair(version, 1, 2)
    storage.commit_repair(version.id, 2, staged, 1, digest)
    leftover = (data_dir / "replica-3" / ".tmp-repair-halfwritten")
    leftover.write_bytes(b"partial")
    storage.set_inspection_progress(inspection.id, "中断前……")

    # 重启：同一 DATA_DIR 重新 init。
    restarted = ObjectStorage(data_dir)
    restarted.init()

    assert not leftover.exists()
    running = restarted.running_inspection()
    assert running is None
    rec = restarted.recent_inspections(1)[0]
    assert rec.id == inspection.id
    assert rec.state == "interrupted"

    checks = restarted.check_replicas(restarted.current_version())
    assert [c.state for c in checks] == ["verified", "verified", "corrupt"]

    app = create_app(data_dir, start_background=False)
    client = TestClient(app)
    with client:
        page = client.get("/")
        assert page.status_code == 200
        assert "可读副本" in page.text and "2/3" in page.text
        assert client.get("/download").content == payload
        client.post("/inspect", data={}, follow_redirects=False)
        status = client.get("/api/status").json()
        assert status["readable"] == 3
        assert status["latest_inspection"]["state"] == "completed"


def test_replace_during_inspection_does_not_touch_new_version(app_env, monkeypatch):
    client, storage = app_env
    old_payload = b"old-version" * 100
    upload(client, old_payload)
    old_version = storage.current_version()

    # 破坏旧版本一份副本。
    replica_path(storage, 2).write_bytes(b"bad-old")

    real_stage = ObjectStorage.stage_repair
    barrier = threading.Event()
    release = threading.Event()

    def paused_stage(self, version, source_slot, target_slot):
        barrier.set()
        assert release.wait(5), "测试同步超时"
        return real_stage(self, version, source_slot, target_slot)

    monkeypatch.setattr(ObjectStorage, "stage_repair", paused_stage)

    inspection = storage.begin_inspection(old_version.id)
    worker = threading.Thread(
        target=run_inspection, args=(storage, inspection.id, old_version.id)
    )
    worker.start()
    assert barrier.wait(5)

    # 巡检暂停时替换为新版本，并破坏一份新副本。
    new_payload = b"new-version" * 100
    upload(client, new_payload)
    new_version = storage.current_version()
    assert new_version.id != old_version.id
    replica_path(storage, 3).write_bytes(b"bad-new")

    release.set()
    worker.join(10)
    assert not worker.is_alive()

    monkeypatch.undo()

    rec = storage.recent_inspections(1)[0]
    assert rec.id == inspection.id
    assert rec.state == "superseded"
    assert "替换" in (rec.summary or "")

    # 新版本未被旧巡检改动：副本 3 仍是坏的，也没有被误报为已修复。
    checks = {c.slot: c for c in storage.check_replicas(new_version)}
    assert checks[3].state == "corrupt"
    assert checks[1].state == "verified" and checks[2].state == "verified"
    status = client.get("/api/status").json()
    assert status["version_id"] == new_version.id
    assert status["readable"] == 2

    # 重新巡检新版本，修复才真正发生。
    inspect(client)
    status = client.get("/api/status").json()
    assert status["version_id"] == new_version.id
    assert status["readable"] == 3
    assert client.get("/download").content == new_payload


def test_stale_inspection_cannot_overwrite_after_replace(app_env, monkeypatch):
    """旧巡检的最终提交发生在替换之后：也不得写入新版本状态。"""
    client, storage = app_env
    upload(client, b"old" * 50)
    old_version = storage.current_version()

    checks = storage.check_replicas(old_version)
    assert storage.is_current(old_version.id)

    upload(client, b"new" * 50)
    new_version = storage.current_version()
    replica_path(storage, 1).write_bytes(b"bad-new")

    # 直接提交一个旧的巡检结果。
    inspection = storage.begin_inspection(new_version.id)
    state = storage.finalize_inspection(
        inspection.id,
        old_version.id,
        "completed",
        "伪造的旧巡检完成",
        checks,
        3,
    )
    assert state == "superseded"

    checks_new = {c.slot: c for c in storage.check_replicas(new_version)}
    assert checks_new[1].state == "corrupt"
    status = client.get("/api/status").json()
    assert status["readable"] == 2
    assert status["latest_inspection"]["state"] == "superseded"


def test_corrupt_endpoint_marks_state(app_env):
    client, storage = app_env
    upload(client, b"abc" * 10)
    version = storage.current_version()
    response = client.post(
        "/replicas/2/corrupt",
        data={"version_id": version.id},
        follow_redirects=False,
    )
    assert response.status_code == 303
    checks = {c.slot: c for c in storage.check_replicas(version)}
    assert checks[2].state == "corrupt"
    assert checks[1].state == "verified"

def test_stale_version_actions_rejected(app_env):
    client, storage = app_env
    upload(client, b"one")
    first = storage.current_version()
    upload(client, b"two")
    second = storage.current_version()

    # 页面带的是旧版本号：下载仍只交付旧版本、且必须与其登记摘要相符。
    response = client.get(f"/download?version_id={first.id}")
    assert response.status_code == 200
    assert response.content == b"one"
    assert response.headers["x-object-version"] == "1"

    # 未知版本不交付任何字节。
    assert client.get("/download?version_id=doesnotexist").status_code == 404

    response = client.post(
        "/replicas/1/corrupt",
        data={"version_id": first.id},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "过期" in unquote(response.headers["location"])

    response = client.post(
        "/inspect", data={"version_id": first.id}, follow_redirects=False
    )
    assert response.status_code == 303
    assert "过期" in unquote(response.headers["location"])
    assert storage.recent_inspections(1) == []
    assert second.id == storage.current_version().id


def test_path_traversal_filename_sanitized(app_env):
    client, storage = app_env
    upload(client, b"x", filename="../../etc/passwd")
    version = storage.current_version()
    assert version.filename == "passwd"
    for slot in (1, 2, 3):
        assert replica_path(storage, slot).is_relative_to(storage.data_dir)


def test_concurrent_inspection_rejected_then_orphan_recovered(app_env):
    client, storage = app_env
    upload(client, b"concurrent")
    version = storage.current_version()
    storage._active_inspections.add(999)  # 占位模拟正在执行的任务
    with pytest.raises(Exception) as exc:
        storage.begin_inspection(version.id)
    assert "已有巡检" in str(exc.value)
    storage._active_inspections.clear()

    # 工作线程已死但 running 没有落库：再次巡检应把它标为中断并继续。
    inspection = storage.begin_inspection(version.id)
    storage.release_inspection(inspection.id)
    orphan = storage.begin_inspection(version.id)
    assert orphan.id != inspection.id
    assert storage.recent_inspections(2)[0].id == orphan.id
    assert storage.recent_inspections(2)[1].state == "interrupted"


def test_old_version_replicas_pruned(app_env):
    client, storage = app_env
    from app.storage import RETAINED_VERSION_REPLICAS

    first_payload = b"v1"
    upload(client, first_payload)
    first = storage.current_version()
    assert client.get(f"/download?version_id={first.id}").content == first_payload

    for i in range(RETAINED_VERSION_REPLICAS):
        upload(client, f"v{i + 2}".encode())

    # 最老的版本副本已清理，但登记摘要仍在；下载被拒绝而不是交付旧字节。
    assert client.get(f"/download?version_id={first.id}").status_code == 409
    assert storage.current_version().seq == RETAINED_VERSION_REPLICAS + 1
    assert client.get("/download").content == f"v{RETAINED_VERSION_REPLICAS + 1}".encode()


def test_repair_ignores_corrupt_source(app_env):
    client, storage = app_env
    payload = b"source-check"
    upload(client, payload)
    version = storage.current_version()
    replica_path(storage, 1).write_bytes(b"bad-1")
    replica_path(storage, 2).write_bytes(b"bad-2")

    # 直接要求用坏副本作为源会失败。
    from app.storage import RepairSourceInvalid

    with pytest.raises(RepairSourceInvalid):
        storage.stage_repair(version, 1, 3)

    inspect(client)
    status = client.get("/api/status").json()
    assert status["readable"] == 3
    assert client.get("/download").content == payload


def block_unlink(monkeypatch, version_id: str, slot: int):
    """让指定版本指定副本文件的删除失败，模拟清退最旧版本时的故障。"""
    real_unlink = Path.unlink
    target_name = f"v-{version_id}.bin"
    target_parent = f"replica-{slot}"

    def guarded(self, missing_ok=False):
        if self.name == target_name and self.parent.name == target_parent:
            raise OSError("simulated prune failure")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", guarded)


def upload_n(client, n: int, start: int = 1) -> dict[int, bytes]:
    payloads = {}
    for i in range(start, start + n):
        data = f"v{i}".encode()
        payloads[i] = data
        upload(client, data)
    return payloads


def test_prune_failure_still_reports_upload_success(app_env, monkeypatch):
    """第 11 次上传清退最旧版本失败：不得误报上传失败，残留不得报成已释放。"""
    from app.storage import PRUNED_REPLICA_NOTE, RETAINED_VERSION_REPLICAS

    client, storage = app_env
    upload_n(client, RETAINED_VERSION_REPLICAS)
    first = storage.list_versions(20)[-1]
    assert first.seq == 1

    block_unlink(monkeypatch, first.id, slot=2)
    new_payload = b"v11"
    response = upload(client, new_payload)
    location = unquote(response.headers["location"])
    assert "level=warning" in location
    assert "已登记版本 #11" in location
    assert "清理未完成" in location
    assert "未改变当前对象" not in location

    # 新版本已登记且可下载。
    current = storage.current_version()
    assert current.seq == RETAINED_VERSION_REPLICAS + 1
    assert client.get("/download").content == new_payload

    # 删除失败的副本仍在磁盘上，且没有被登记成已清理；其余副本如实登记。
    stuck = replica_path(storage, 2, first.id)
    assert stuck.exists()
    rows = storage.replica_rows(first.id)
    assert rows[2]["note"] != PRUNED_REPLICA_NOTE
    assert rows[1]["note"] == PRUNED_REPLICA_NOTE
    assert rows[3]["note"] == PRUNED_REPLICA_NOTE

    # 操作日志只有清理警告，没有把版本 #1 报成已释放的成功记录。
    events = storage.recent_events(30)
    prune_info = [
        e for e in events if e["kind"] == "prune" and e["level"] == "info"
    ]
    assert any(
        e["kind"] == "prune" and e["level"] == "warning" for e in events
    )
    assert not any(f"#{first.seq} " in e["message"] for e in prune_info)

    # 首页持续提示清理未完成（刷新后仍可见）。
    page = client.get("/").text
    assert "清理未完成" in page
    assert f"#{first.seq}" in page

    # 残留副本仍属登记版本：旧版本按所报版本号下载时只能交付真实字节。
    download = client.get(f"/download?version_id={first.id}")
    assert download.status_code == 200
    assert download.content == b"v1"


def test_prune_resumes_after_restart(app_env, monkeypatch):
    """清理失败后在重启时接续重试；障碍消除后重试成功。"""
    from app.storage import RETAINED_VERSION_REPLICAS

    client, storage = app_env
    upload_n(client, RETAINED_VERSION_REPLICAS)
    first = storage.list_versions(20)[-1]

    block_unlink(monkeypatch, first.id, slot=2)
    upload(client, b"v11")
    stuck = replica_path(storage, 2, first.id)
    assert stuck.exists()
    assert storage.pending_prune_versions() == [first.seq]

    # 重启（删除障碍仍在）：启动时重试清理，仍失败，仍如实挂起。
    restarted = ObjectStorage(storage.data_dir)
    restarted.init()
    assert stuck.exists()
    assert restarted.pending_prune_versions() == [first.seq]
    events = restarted.recent_events(30)
    assert any(
        e["kind"] == "prune" and e["level"] == "warning" for e in events
    )

    # 再次重启（障碍已消除）：接续清理成功，残留文件被释放。
    monkeypatch.undo()
    recovered = ObjectStorage(storage.data_dir)
    recovered.init()
    assert not stuck.exists()
    assert recovered.pending_prune_versions() == []
    rows = recovered.replica_rows(first.id)
    assert all(row["note"] == "历史版本副本已按保留策略清理" for row in rows.values())

    # 旧版本副本已真正释放：下载拒绝；当前版本不受影响。
    assert client.get(f"/download?version_id={first.id}").status_code == 409
    assert client.get("/download").content == b"v11"
    # 挂起横幅已消失（历史日志中的清理警告记录仍保留，属正常）。
    assert "仍占用磁盘" not in client.get("/").text


def test_prune_recomputes_retention_window(app_env, monkeypatch):
    """清理未完成期间再上传：按最新的最近 10 版重算待删集合，不误删保留版本。"""
    from app.storage import RETAINED_VERSION_REPLICAS

    client, storage = app_env
    upload_n(client, RETAINED_VERSION_REPLICAS)
    versions = {v.seq: v for v in storage.list_versions(20)}
    first, second, third = versions[1], versions[2], versions[3]

    # 第 11 次上传：v1 副本 2 删除失败，v1 残留。
    block_unlink(monkeypatch, first.id, slot=2)
    upload(client, b"v11")
    assert replica_path(storage, 2, first.id).exists()

    # 清理未完成期间上传第 12 版：重算后 v1、v2 都超出最新保留窗口。
    monkeypatch.undo()
    response = upload(client, b"v12")
    location = unquote(response.headers["location"])
    assert "level=success" in location
    assert "清理未完成" not in location

    # v1、v2 的副本全部释放；v3 起仍在保留窗口内，不得误删。
    for slot in (1, 2, 3):
        assert not replica_path(storage, slot, first.id).exists()
        assert not replica_path(storage, slot, second.id).exists()
        assert replica_path(storage, slot, third.id).exists()
    assert client.get(f"/download?version_id={first.id}").status_code == 409
    assert client.get(f"/download?version_id={second.id}").status_code == 409
    assert client.get(f"/download?version_id={third.id}").content == b"v3"
    assert storage.pending_prune_versions() == []
    assert client.get("/download").content == b"v12"


def test_failed_upload_keeps_current_version(app_env, monkeypatch):
    """登记前失败（副本写入异常）：保持原当前版可下载，不产生新版本。"""
    client, storage = app_env
    upload(client, b"stable-v1")
    first = storage.current_version()

    def failing_write(destination, data):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(
        ObjectStorage, "_atomic_write", staticmethod(failing_write)
    )
    response = upload(client, b"should-not-register")
    location = unquote(response.headers["location"])
    assert "level=error" in location
    assert "未改变当前对象" in location

    assert storage.current_version().id == first.id
    assert len(storage.list_versions(10)) == 1
    assert client.get("/download").content == b"stable-v1"


def test_prune_exception_still_reports_upload_success(app_env, monkeypatch):
    """清理任务自身异常（如数据库故障）也不得误报为上传失败。"""
    client, storage = app_env

    def broken_prune():
        raise RuntimeError("simulated prune crash")

    monkeypatch.setattr(ObjectStorage, "prune_old_versions", broken_prune)
    response = upload(client, b"prune-crash")
    location = unquote(response.headers["location"])
    assert "level=warning" in location
    assert "已登记版本 #1" in location
    assert "清理未完成" in location
    assert "未改变当前对象" not in location
    assert client.get("/download").content == b"prune-crash"


def test_prune_report_counts_and_idempotent(tmp_path: Path):
    """清理报告如实计数；重复调用幂等，不重复记日志。"""
    from app.storage import RETAINED_VERSION_REPLICAS

    storage = ObjectStorage(tmp_path / "data")
    storage.init()
    for i in range(RETAINED_VERSION_REPLICAS + 2):
        storage.create_version(f"v{i}".encode(), f"f{i}.bin")

    report = storage.prune_old_versions()
    assert report.ok
    assert report.removed_files == 6  # v1、v2 各 3 个副本
    assert report.freed_versions == (1, 2)
    assert report.failures == ()
    assert storage.pending_prune_versions() == []
    prune_logs = [
        e for e in storage.recent_events(50) if e["kind"] == "prune"
    ]
    assert len(prune_logs) == 2

    again = storage.prune_old_versions()
    assert again.ok
    assert again.removed_files == 0
    assert again.freed_versions == ()
    prune_logs_after = [
        e for e in storage.recent_events(50) if e["kind"] == "prune"
    ]
    assert len(prune_logs_after) == 2  # 没有重复记日志
