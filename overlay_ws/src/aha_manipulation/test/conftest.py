import pytest
import rclpy
from rclpy.context import Context


@pytest.fixture(scope="session")
def rclpy_context() -> Context:
    context = rclpy.create_context()
    rclpy.init(context=context)
    yield context
    if rclpy.ok(context=context):
        rclpy.shutdown(context=context)
