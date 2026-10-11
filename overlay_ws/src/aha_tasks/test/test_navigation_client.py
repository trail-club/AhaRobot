"""Exercise the real ROS action protocol using a stationary Nav2 test double."""

from pathlib import Path
import sys
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
from lifecycle_msgs.srv import GetState  # noqa: E402
from nav2_msgs.action import NavigateToPose  # noqa: E402
from rclpy.action import ActionServer, CancelResponse, GoalResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aha_tasks.navigation import Navigation  # noqa: E402


@pytest.fixture
def server():
    rclpy.init()
    node = Node("test_nav2")
    client_node = Node("test_task_client")
    state = {
        "mode": "success",
        "active": True,
        "goals": [],
        "cancelled": threading.Event(),
    }
    group = ReentrantCallbackGroup()

    def get_state(request, response):
        response.current_state.id = 3 if state["active"] else 2
        return response

    def execute(goal):
        state["goals"].append(goal.request.pose)
        if state["mode"] == "timeout":
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not goal.is_cancel_requested:
                time.sleep(0.01)
            if goal.is_cancel_requested:
                goal.canceled()
                state["cancelled"].set()
            else:
                goal.abort()
        elif state["mode"] == "abort":
            goal.abort()
        else:
            goal.succeed()
        return NavigateToPose.Result()

    def accept_goal(request):
        time.sleep(state.get("accept_delay", 0.0))
        return GoalResponse.REJECT if state["mode"] == "reject" else GoalResponse.ACCEPT

    service = node.create_service(
        GetState, "/bt_navigator/get_state", get_state, callback_group=group
    )
    action = ActionServer(
        node,
        NavigateToPose,
        "/navigate_to_pose",
        execute,
        callback_group=group,
        goal_callback=accept_goal,
        cancel_callback=lambda goal: CancelResponse.ACCEPT,
    )
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    navigation = Navigation(client_node, startup_timeout=5, goal_timeout=0.5)
    yield navigation, state
    navigation.cancel()
    executor.shutdown()
    thread.join(timeout=5)
    action.destroy()
    node.destroy_service(service)
    node.destroy_node()
    client_node.destroy_node()
    rclpy.shutdown()


@pytest.mark.parametrize(
    "mode,success", [("success", True), ("reject", False), ("abort", False)]
)
def test_action_results(server, mode, success):
    navigation, state = server
    state["mode"] = mode
    navigation.ready()
    assert navigation.navigate([-2.0, 1.5, 1.0]) is success
    if mode != "reject":
        pose = state["goals"][0]
        assert pose.header.frame_id == "map"
        assert pose.pose.position.x == -2.0
        assert pose.pose.orientation.z == pytest.approx(0.47942554)


def test_timeout_cancels_active_goal(server):
    navigation, state = server
    state["mode"] = "timeout"
    navigation.ready()
    with pytest.raises(TimeoutError):
        navigation.navigate([0, 0, 0])
    navigation.cancel()
    assert state["cancelled"].is_set()


def test_inactive_server_is_not_ready(server):
    navigation, state = server
    state["active"] = False
    navigation.startup_timeout = 0.3
    with pytest.raises(TimeoutError):
        navigation.ready()
    assert not state["goals"]


def test_late_acceptance_is_cancelled(server):
    navigation, state = server
    state.update(mode="timeout", accept_delay=0.2)
    navigation.ready()
    wait = navigation.wait
    navigation.wait = lambda future, timeout: wait(future, 0.01)
    try:
        with pytest.raises(TimeoutError):
            navigation.navigate([0, 0, 0])
    finally:
        navigation.wait = wait
    navigation.cancel()
    assert state["cancelled"].is_set()
