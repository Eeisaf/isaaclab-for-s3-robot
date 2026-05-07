from pxr import Usd, UsdGeom, UsdPhysics
import sys

def get_joints(usd_path):
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print("Failed to open stage")
        return
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) or prim.GetTypeName() in ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsJoint"]:
            joints.append(prim.GetName())
    print("Joints found:", joints)

if __name__ == "__main__":
    get_joints(sys.argv[1])
