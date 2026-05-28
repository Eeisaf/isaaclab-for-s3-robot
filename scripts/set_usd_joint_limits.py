from __future__ import annotations

"""Set revolute joint limits in a USD file.

Run this with the Isaac Sim/Lab Python environment because the normal project
Python may not include the pxr USD bindings.
"""

import argparse
import math
import os

from pxr import Sdf, Usd, UsdPhysics


def _find_joint_prim(stage: Usd.Stage, joint_name: str) -> Usd.Prim:
    matches = []
    for prim in stage.Traverse():
        if prim.GetName() == joint_name:
            matches.append(prim)

    if not matches:
        raise RuntimeError(f"Joint {joint_name!r} was not found in {stage.GetRootLayer().realPath}")
    if len(matches) > 1:
        paths = [str(prim.GetPath()) for prim in matches]
        raise RuntimeError(f"Joint name {joint_name!r} is ambiguous: {paths}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="Set USD revolute joint lower/upper limits.")
    parser.add_argument(
        "--usd",
        default="source/DEMO_learn/trunk_robot/configuration/trunk_robot_physics.usd",
        help="USD layer containing the physics joint definitions.",
    )
    parser.add_argument("--joint", default="trunk_joint3", help="Joint prim name to modify.")
    parser.add_argument("--lower_rad", type=float, default=-math.pi, help="Lower joint limit in radians.")
    parser.add_argument("--upper_rad", type=float, default=math.pi, help="Upper joint limit in radians.")
    parser.add_argument(
        "--degrees",
        action="store_true",
        help="Interpret lower/upper values as degrees instead of radians.",
    )
    args = parser.parse_args()

    usd_path = os.path.abspath(args.usd)
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {usd_path}")

    joint_prim = _find_joint_prim(stage, args.joint)
    revolute_joint = UsdPhysics.RevoluteJoint(joint_prim)
    if not revolute_joint:
        raise RuntimeError(f"Prim {joint_prim.GetPath()} is not a UsdPhysics.RevoluteJoint")

    lower = args.lower_rad if args.degrees else math.degrees(args.lower_rad)
    upper = args.upper_rad if args.degrees else math.degrees(args.upper_rad)
    lower_attr = revolute_joint.GetLowerLimitAttr()
    upper_attr = revolute_joint.GetUpperLimitAttr()
    old_lower = lower_attr.Get()
    old_upper = upper_attr.Get()

    lower_attr.Set(lower)
    upper_attr.Set(upper)

    stage.GetRootLayer().Save()

    print(f"[INFO] USD: {usd_path}")
    print(f"[INFO] Joint: {joint_prim.GetPath()}")
    print(f"[INFO] lowerLimit: {old_lower} deg -> {lower} deg")
    print(f"[INFO] upperLimit: {old_upper} deg -> {upper} deg")


if __name__ == "__main__":
    main()
