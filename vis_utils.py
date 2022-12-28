import matplotlib.pyplot as plt  
# from pylab import *
import os
from os.path import join as pjoin
from utils import ensure_dirs
import numpy as np

def set_axes_equal(ax, limits=None, labels=True):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    ax.set_box_aspect([1, 1, 1])
    if limits is None:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    else:
        x_limits, y_limits, z_limits = limits
        ax.set_xlim3d([x_limits[0], x_limits[1]])
        ax.set_ylim3d([y_limits[0], y_limits[1]])
        ax.set_zlim3d([z_limits[0], z_limits[1]])

    if labels:
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')


def plot3d_pts(pts, pts_name=None, s=2, dpi=100, title_name=None,
               color_channel=None, colorbar=False, color_range=None, color_raw=None,
               bcm=None, markers=None, arrows=None,
               puttext=None, view_angle=None, limits=None,
               save_fig=False, save_folder=None,
               save_name=None,
               axis_off=False, show_fig=True,
               shape=None, plot_sphere_radius=None):
    if pts_name is None:
        pts_name = [["" for _ in pt_groups] for pt_groups in pts]
        use_pts_name = False
    else:
        use_pts_name = True
    num = len(pts)
    if shape is None:
        shape = (1, num)
    fig = plt.figure(figsize=(4 * shape[1], 4 * shape[0]), dpi=dpi)
    cmap = plt.cm.jet
    top = plt.cm.get_cmap('Oranges_r', 128)
    bottom = plt.cm.get_cmap('Blues', 128)

    colors = np.vstack((top(np.linspace(0, 1, 30)),
                        bottom(np.linspace(0, 1, 30))))
    all_poss = ['o', 'o', 'o', '.', 'o', '*', '.', 'o', 'v', '^', '>', '<', 's', 'p', '*', 'h', 'H', 'D', 'd', '1', '',
                '']

    if view_angle is None or isinstance(view_angle, tuple):
        view_angle = [view_angle for _ in range(num)]

    bcm_list = bcm
    markers_list = markers
    arrows_list = arrows
    color_channel_list = color_channel
    text_list = puttext
    color_range_list = color_range

    for m in range(num):
        ax = plt.subplot(shape[0], shape[1], m + 1, projection='3d')
        if view_angle[m] is None:
            ax.view_init(elev=30, azim=-40)
        else:
            ax.view_init(azim=view_angle[m][0], elev=view_angle[m][1])

        color_channel = None if color_channel_list is None else color_channel_list[m]
        color_range = None if color_range_list is None else color_range_list[m]
        cur_color = None if color_raw is None else color_raw[m]
        for n in range(len(pts[m])):
            if cur_color is not None:
                ax.scatter(pts[m][n][:, 0], pts[m][n][:, 1], pts[m][n][:, 2], marker=all_poss[0], s=s,
                           c=cur_color[n], alpha=1.0)
            elif color_channel is None:
                ax.scatter(pts[m][n][:, 0], pts[m][n][:, 1], pts[m][n][:, 2], marker=all_poss[0], s=s,
                           cmap=colors[n], label=pts_name[m][n], alpha=1.0)
            else:
                if colorbar:
                    rgb_encoded = color_channel[n]
                else:
                    rgb_encoded = (color_channel[n] - np.amin(color_channel[n], axis=0, keepdims=True)) / np.array(
                        np.amax(color_channel[n], axis=0, keepdims=True) - np.amin(color_channel[n], axis=0,
                                                                                   keepdims=True))
                if len(pts[m]) == 3 and n == 2:
                    p = ax.scatter(pts[m][n][:, 0], pts[m][n][:, 1], pts[m][n][:, 2], marker=all_poss[4], s=s,
                                   c=rgb_encoded, label=pts_name[m][n])
                else:
                    p = ax.scatter(pts[m][n][:, 0], pts[m][n][:, 1], pts[m][n][:, 2], marker=all_poss[0], s=s,
                                   c=rgb_encoded, label=pts_name[m][n])
                if colorbar:
                    if color_range is not None:
                        p.set_clim(*color_range)
                    fig.colorbar(p)

        if axis_off:
            plt.axis('off')
            plt.grid('off')
        else:
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

        if title_name is not None:
            plt.title(title_name[m], fontsize=16)

        if use_pts_name:
            plt.legend(loc='lower left')

        if bcm_list is not None:
            bcm = bcm_list[m]
            for j in range(len(bcm)):
                ax.plot3D([bcm[j][0][0], bcm[j][2][0], bcm[j][6][0], bcm[j][4][0], bcm[j][0][0]], \
                          [bcm[j][0][1], bcm[j][2][1], bcm[j][6][1], bcm[j][4][1], bcm[j][0][1]], \
                          [bcm[j][0][2], bcm[j][2][2], bcm[j][6][2], bcm[j][4][2], bcm[j][0][2]], 'blue')

                ax.plot3D([bcm[j][1][0], bcm[j][3][0], bcm[j][7][0], bcm[j][5][0], bcm[j][1][0]], \
                          [bcm[j][1][1], bcm[j][3][1], bcm[j][7][1], bcm[j][5][1], bcm[j][1][1]], \
                          [bcm[j][1][2], bcm[j][3][2], bcm[j][7][2], bcm[j][5][2], bcm[j][1][2]], 'gray')

                for pair in [[0, 1], [2, 3], [4, 5], [6, 7]]:
                    ax.plot3D([bcm[j][pair[0]][0], bcm[j][pair[1]][0]], \
                              [bcm[j][pair[0]][1], bcm[j][pair[1]][1]], \
                              [bcm[j][pair[0]][2], bcm[j][pair[1]][2]], 'red')

        if markers_list is not None:
            markers = markers_list[m]
            for marker in markers:
                ax.plot3D([marker[0]], [marker[1]], [marker[2]], 'ro')

        if arrows_list is not None:
            arrows = arrows_list[m]
            for arrow in arrows:  # (x, y, z, dx, dy, dz)
                # ax.arrow(arrow[0], arrow[1], arrow[2], arrow[3], arrow[4], arrow[5], width=0.1)
                ax.quiver(arrow[0], arrow[1], arrow[2], arrow[3], arrow[4], arrow[5], length=1.0, color='red')

        if text_list is not None:
            text = text_list[m]
            if text is not None:
                ax.text2D(0.05, 0.65, text, transform=ax.transAxes, color='blue', fontsize=10)
        if limits is not None:
            cur_limits = limits[m]
        else:
            cur_limits = [[-1, 1], [-1, 1], [-1, 1]]
        set_axes_equal(ax, limits=cur_limits)

        if plot_sphere_radius is not None:
            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
            x = np.cos(u) * np.sin(v) * plot_sphere_radius
            y = np.sin(u) * np.sin(v) * plot_sphere_radius
            z = np.cos(v) * plot_sphere_radius
            ax.plot_wireframe(x, y, z, color="r", alpha=0.1)

    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_fig:
        safe_save_fig(fig, pjoin(save_folder, f'{save_name}.png'))
    plt.close()


def hand_vis(pts, init_kp, pred_kp, gt_kp, add_obj=None, show_fig=False, save_fig=False, save_folder=None,save_name=None,titles=(-0.01,-0.01)):
    color = ['red', 'blue', 'green', 'orange', 'grey']

    def draw_kp(ax, pc):
        for i in range(0, 5):
            x0 = pc[0, 0]
            y0 = pc[0, 1]
            z0 = pc[0, 2]
            x1 = pc[i * 4 + 1, 0]
            y1 = pc[i * 4 + 1, 1]
            z1 = pc[i * 4 + 1, 2]
            line01_x = np.linspace(x0, x1, num=100)
            line01_y = np.linspace(y0, y1, num=100)
            line01_z = np.linspace(z0, z1, num=100)

            sc = ax.scatter(line01_x, line01_y, line01_z, s=2, c=color[0], alpha=0.5)

        for i in range(0, 5):
            x0 = pc[i * 4 + 1, 0]
            y0 = pc[i * 4 + 1, 1]
            z0 = pc[i * 4 + 1, 2]
            x1 = pc[i * 4 + 2, 0]
            y1 = pc[i * 4 + 2, 1]
            z1 = pc[i * 4 + 2, 2]
            line01_x = np.linspace(x0, x1, num=100)
            line01_y = np.linspace(y0, y1, num=100)
            line01_z = np.linspace(z0, z1, num=100)

            sc = ax.scatter(line01_x, line01_y, line01_z,  s=2, c= color[1], alpha=0.5)

        for i in range(0, 5):
            x0 = pc[i * 4 + 2, 0]
            y0 = pc[i * 4 + 2, 1]
            z0 = pc[i * 4 + 2, 2]
            x1 = pc[i * 4 + 3, 0]
            y1 = pc[i * 4 + 3, 1]
            z1 = pc[i * 4 + 3, 2]
            line01_x = np.linspace(x0, x1, num=100)
            line01_y = np.linspace(y0, y1, num=100)
            line01_z = np.linspace(z0, z1, num=100)

            sc = ax.scatter(line01_x, line01_y, line01_z, s=2, c= color[2], alpha=0.5)

        for i in range(0, 5):
            x0 = pc[i * 4 + 3, 0]
            y0 = pc[i * 4 + 3, 1]
            z0 = pc[i * 4 + 3, 2]
            x1 = pc[i * 4 + 4, 0]
            y1 = pc[i * 4 + 4, 1]
            z1 = pc[i * 4 + 4, 2]
            line01_x = np.linspace(x0, x1, num=100)
            line01_y = np.linspace(y0, y1, num=100)
            line01_z = np.linspace(z0, z1, num=100)

            sc = ax.scatter(line01_x, line01_y, line01_z, s=2, c= color[3], alpha=0.5)
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        sc = ax.scatter(x, y, z, marker='*', s=1, c=color[4], alpha=0.5)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1,3,1, projection='3d')
    ax.view_init(elev=30, azim=-40)
    draw_kp(ax, init_kp)
    ax.scatter(pts[:, 0],pts[:, 1],pts[:, 2],c='grey')
    if add_obj is not None:
        ax.scatter(add_obj[:, 0], add_obj[:, 1], add_obj[:, 2], c='purple')

    ax.title.set_text('init %.2f'%(titles[0]*100))

    ax = fig.add_subplot(1,3,2, projection='3d')
    ax.view_init(elev=30, azim=-40)
    draw_kp(ax, pred_kp)
    ax.scatter(pts[:, 0],pts[:, 1],pts[:, 2], c='grey')
    if add_obj is not None:
        ax.scatter(add_obj[:, 0], add_obj[:, 1], add_obj[:, 2], c='purple')
    ax.title.set_text('pred %.2f'%(titles[1]*100))

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.view_init(elev=30, azim=-40)
    draw_kp(ax, gt_kp)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='grey')
    if add_obj is not None:
        ax.scatter(add_obj[:, 0], add_obj[:, 1], add_obj[:, 2], c='purple')
    ax.title.set_text('gt')

    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_fig:
        safe_save_fig(fig, pjoin(save_folder, f'{save_name}.png'))

def safe_save_fig(fig, save_path):
    ensure_dirs(os.path.dirname(save_path))
    fig.savefig(save_path)


