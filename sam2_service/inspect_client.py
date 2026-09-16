#!/usr/bin/env python3
"""
inspect_client.py -- call the perception pipeline from outside Isaac Sim.

This is the external entry point for the integrated system: a task
planner, a locomotion node, or a test script sends ONE "inspect" command
with the target object name (any phrase -- the classifier is
open-vocabulary) and optionally a goal pose, then polls "status" until
the state is "done" and reads the .pcd path from the reply.

Needs only pyzmq (works from the ROS1 container too, thanks to
--network host). Run examples (host or container):

    # full autonomous chain, goal pose in world frame (metres, degrees)
    python3 inspect_client.py inspect --target "cereal box" --x 2.0 --y 4.8 --theta 90 --wait

    # goal pose relative to where the robot started
    python3 inspect_client.py inspect --target bottle --x 1.5 --y 0 --theta 0 --frame start --wait

    # target only, no move (robot already at the table)
    python3 inspect_client.py inspect --target mug --wait

    python3 inspect_client.py status
    python3 inspect_client.py abort

As a library:

    from inspect_client import PerceptionClient
    pc = PerceptionClient("tcp://131.220.7.222:5556")
    pc.inspect("bottle", goal={"x": 2.0, "y": 4.8, "theta_deg": 90})
    result = pc.wait_done(timeout_s=180)     # -> task dict with "pcd" path
"""
import sys
import json
import time
import argparse
import zmq

DEFAULT_ADDR = "tcp://131.220.7.222:5556"
TERMINAL_STATES = {"done", "failed", "idle"}


class PerceptionClient:
    def __init__(self, addr=DEFAULT_ADDR, timeout_ms=5000):
        self.addr = addr
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context.instance()
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.addr)

    def _call(self, req):
        self.sock.send_multipart([json.dumps(req).encode("utf-8")])
        try:
            parts = self.sock.recv_multipart()
        except zmq.Again:
            # a REQ socket is unusable after a timeout -- rebuild it
            self.sock.close(linger=0)
            self._connect()
            raise TimeoutError(f"no reply from {self.addr} within {self.timeout_ms} ms "
                               f"(is the native GUI running inside Isaac Sim?)")
        return json.loads(parts[0].decode("utf-8"))

    def inspect(self, target, goal=None, auto_plan=True, auto_fuse=True, teleport_to_goal=True):
        req = {"cmd": "inspect", "target": target, "auto_plan": auto_plan, "auto_fuse": auto_fuse,
               "teleport_to_goal": teleport_to_goal}
        if goal is not None:
            req["goal"] = goal
        return self._call(req)

    def status(self):
        return self._call({"cmd": "status"})

    def abort(self):
        return self._call({"cmd": "abort"})

    def wait_done(self, timeout_s=300, poll_s=1.0, verbose=True):
        """Poll status until the task reaches done/failed (or idle after an abort)."""
        t0 = time.time()
        last = None
        while time.time() - t0 < timeout_s:
            st = self.status()
            task = st.get("task", {})
            line = f"{task.get('state')}: {task.get('message', '')}  views={st.get('views')}"
            if verbose and line != last:
                print(f"[{time.time() - t0:5.1f}s] {line}")
                last = line
            if task.get("state") in TERMINAL_STATES:
                return task
            time.sleep(poll_s)
        raise TimeoutError(f"task not finished after {timeout_s}s (last: {last})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inspect", "status", "abort"])
    ap.add_argument("--addr", default=DEFAULT_ADDR)
    ap.add_argument("--target", help="object name to look for (any phrase)")
    ap.add_argument("--x", type=float); ap.add_argument("--y", type=float)
    ap.add_argument("--theta", type=float, default=0.0, help="heading in degrees")
    ap.add_argument("--frame", choices=["world", "start"], default="world")
    ap.add_argument("--no-move", action="store_true", help="don't teleport to the goal (locomotion does it)")
    ap.add_argument("--no-auto", action="store_true", help="only search+lock; leave plan/fuse to the GUI buttons")
    ap.add_argument("--wait", action="store_true", help="poll status until done and print the .pcd path")
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()

    pc = PerceptionClient(a.addr)
    if a.cmd == "status":
        print(json.dumps(pc.status(), indent=2)); return
    if a.cmd == "abort":
        print(json.dumps(pc.abort(), indent=2)); return
    if not a.target:
        sys.exit("--target is required for inspect")
    goal = None
    if a.x is not None and a.y is not None:
        goal = {"x": a.x, "y": a.y, "theta_deg": a.theta, "frame": a.frame}
    r = pc.inspect(a.target, goal=goal, auto_plan=not a.no_auto, auto_fuse=not a.no_auto,
                   teleport_to_goal=not a.no_move)
    print(json.dumps(r, indent=2))
    if not r.get("ok"):
        sys.exit(1)
    if a.wait:
        task = pc.wait_done(timeout_s=a.timeout)
        print(json.dumps(task, indent=2))
        if task.get("state") != "done":
            sys.exit(2)
        print(f"\nPOINT CLOUD: {task.get('pcd')}")


if __name__ == "__main__":
    main()
