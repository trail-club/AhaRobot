"""Offline tests: extract timed() without importing ROS or issuing hardware commands."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


class TimedMotionTests(unittest.TestCase):
    def setup_motion(self, stale=False, displacement=0.0):
        clock = SimpleNamespace(now=0.0)
        sent = []
        node = SimpleNamespace(odom_pose=(0.0, 0.0, 0.0), odom_yaw_unwrapped=0.0,
                               last_odom_wall=-1.0 if stale else 0.0, stopped=False)
        node.cmd_pub = SimpleNamespace(publish=lambda cmd: sent.append(clock.now))

        def stop():
            node.stopped = True
            clock.now += 0.2

        def spin_once(_node, timeout_sec):
            clock.now += timeout_sec
            if not stale:
                node.last_odom_wall = clock.now

        node.stop = stop
        source = Path(__file__).with_name('final_motion_evaluation.py').read_text()
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'timed')
        namespace = dict(node=node, time=SimpleNamespace(monotonic=lambda: clock.now),
                         math=math, rclpy=SimpleNamespace(spin_once=spin_once),
                         Twist=lambda: SimpleNamespace(linear=SimpleNamespace(x=0.0),
                                                       angular=SimpleNamespace(z=0.0)),
                         sm=SimpleNamespace(relative_pose=lambda _a, _b: (displacement, 0.0, 0.0)),
                         report={'commands': []})
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<timed>', 'exec'), namespace)
        return namespace['timed'], node, sent

    def test_duration_ends_even_without_encoder_progress(self):
        run, node, sent = self.setup_motion()
        result = run(0.03, 0.0, 1.0)
        self.assertTrue(node.stopped)
        self.assertGreater(len(sent), 40)
        self.assertLess(sent[-1], 1.02)
        self.assertGreaterEqual(result['publish_and_stop_duration_s'], 1.2)

    def test_stale_odometry_stops_before_nonzero_publish(self):
        run, node, sent = self.setup_motion(stale=True)
        with self.assertRaisesRegex(RuntimeError, 'Stale'):
            run(0.03, 0.0, 1.0)
        self.assertTrue(node.stopped)
        self.assertEqual(sent, [])

    def test_excess_displacement_stops_before_nonzero_publish(self):
        run, node, sent = self.setup_motion(displacement=0.1)
        with self.assertRaisesRegex(RuntimeError, 'displacement'):
            run(0.03, 0.0, 1.0)
        self.assertTrue(node.stopped)
        self.assertEqual(sent, [])


if __name__ == '__main__':
    unittest.main()
