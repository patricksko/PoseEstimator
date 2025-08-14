import numpy as np
import matplotlib.pyplot as plt

def to_homogeneous(R_flat, t):
    """Convert flat R (list of 9) and t (list of 3) to 4×4 homogeneous transformation matrix."""
    R = np.array(R_flat).reshape(3, 3)
    t = np.array(t).reshape(3, 1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t[:, 0]
    return T

def plot_frame(ax, T, label='', length=100, alpha=1.0):
    """Plot coordinate frame from transformation matrix."""
    origin = T[:3, 3]
    ax.quiver(*origin, *T[:3, 0], color='r', length=length, normalize=True, alpha=alpha)
    ax.quiver(*origin, *T[:3, 1], color='g', length=length, normalize=True, alpha=alpha)
    ax.quiver(*origin, *T[:3, 2], color='b', length=length, normalize=True, alpha=alpha)
    if label:
        ax.text(*origin, label, fontsize=8)

# ------------------ 1) From BOP: model → camera ------------------
cam_R_m2c = [
    0.149011492729187, 0.9648433923721313, 0.21650098264217377,
    -0.5316080451011658, 0.26277977228164673, -0.8051953911781311,
    -0.8337795734405518, 0.004889651667326689, 0.5520758032798767
]
cam_t_m2c = [-0.5411790013313293, 2.012904644012451, 872.8438720703125]  # mm

T_m2c = to_homogeneous(cam_R_m2c, cam_t_m2c)

# ------------------ 2) From camera JSON: world → camera ------------------
cam_R_w2c = [
    -0.1331634819507599, -0.9671581983566284, -0.21650086343288422,
    -0.5845581293106079, -0.09976007789373398, 0.8051955103874207,
    -0.8003495335578918, 0.2337799370288849, -0.5520758032798767
]
cam_t_w2c = [-244.3331298828125, -195.26478576660156, 726.9111938476562]  # mm

T_w2c = to_homogeneous(cam_R_w2c, cam_t_w2c)
T_c2w = np.linalg.inv(T_w2c)  # camera → world

# ------------------ 3) From BlenderProc: object → world (meters) ------------------
obj2w_data = [
    [0.0009582280763424933, -0.00028600532095879316, 4.4745568361648225e-11, -264.5815908908844],
    [-0.00028600532095879316, -0.0009582280763424933, -1.045891429019008e-10, -221.34965658187866],
    [7.278951053013571e-11, 8.742277735063197e-11, -0.0010000000474974513, 25.500008836388588],
    [0.0, 0.0, 0.0, 1.0]
]
T_o2w = np.array(obj2w_data).copy()
# T_o2w[:3, 3] *= 1000  # convert meters → mm

# ------------------ 4) Derived: model → world from cam + bop ------------------
T_m2w = T_c2w @ T_m2c  # m→w via m→c→w

# ------------------ Plot all frames ------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_frame(ax, T_o2w, 'o2w (BlenderProc)', length=50)
plot_frame(ax, T_m2w, 'm2w (BOP via cam)', length=50)
#plot_frame(ax, T_c2w, 'c2w (camera)', length=50)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Pose Consistency Check')
# ax.set_xlim(264,265)
# ax.set_ylim(221, 222)
# ax.set_zlim(25, 26)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=30, azim=45)
plt.show()
