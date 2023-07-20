import imageio
from numpy import matrix, array
import numpy as np


def overlay(icon, background):
    return np.where(np.equal(icon, np.full(3, 255, dtype=np.uint8)),
                    background,
                    icon)


cpo_fn = "trajectories/Traj-MidObstac234904.txt"
revel_fn = "obstacle_traj.txt"
static_fn = "obstacle_traj_static.txt"
cpo_traj = []
with open(cpo_fn, 'r') as cpo_file:
    for line in cpo_file:
        cpo_traj.append(eval(line))
revel_traj = []
with open(revel_fn, 'r') as revel_file:
    step = ""
    for line in revel_file:
        if line[0].isspace():
            step += line
        else:
            if len(step) > 0:
                tmp = eval(step)
                revel_traj.append(np.squeeze(array(tmp[0][:4])))
            step = line
    tmp = eval(step)
    revel_traj.append(np.squeeze(array(tmp[0][:4])))
static_traj = []
with open(static_fn, 'r') as static_file:
    step = ""
    for line in static_file:
        if line[0].isspace():
            step += line
        else:
            if len(step) > 0:
                tmp = eval(step)
                static_traj.append(np.squeeze(array(tmp[0][:4])))
            step = line
    tmp = eval(step)
    static_traj.append(np.squeeze(array(tmp[0][:4])))

cpo_next = [cpo_traj[-1][0] + 0.01 * cpo_traj[-1][2],
            cpo_traj[-1][1] + 0.01 * cpo_traj[-1][3],
            cpo_traj[-1][2],
            cpo_traj[-1][3]]
cpo_traj.append(cpo_next)
cpo_traj = array(cpo_traj)
revel_next = [revel_traj[-1][0] + 0.01 * revel_traj[-1][2],
              revel_traj[-1][1] + 0.01 * revel_traj[-1][3],
              revel_traj[-1][2],
              revel_traj[-1][3]]
revel_traj.append(revel_next)
revel_traj = array(revel_traj)
static_next = [static_traj[-1][0] + 0.01 * static_traj[-1][2],
               static_traj[-1][1] + 0.01 * static_traj[-1][3],
               static_traj[-1][2],
               static_traj[-1][3]]
static_traj.append(static_next)
static_traj = array(static_traj)

width = 400
height = 300
frames = 200

dilation = 1

min_x = min(np.min(revel_traj[:frames * dilation, 0]),
            np.min(cpo_traj[:frames * dilation, 0]),
            np.min(static_traj[:frames * dilation, 0]))
max_x = max(np.max(revel_traj[:frames * dilation, 0]),
            np.max(cpo_traj[:frames * dilation, 0]),
            np.max(static_traj[:frames * dilation, 0]))
min_y = min(np.min(revel_traj[:frames * dilation, 1]),
            np.min(cpo_traj[:frames * dilation, 1]),
            np.min(static_traj[:frames * dilation, 1]))
max_y = max(np.max(revel_traj[:frames * dilation, 1]),
            np.max(cpo_traj[:frames * dilation, 1]),
            np.max(static_traj[:frames * dilation, 1]))

buffer = max((max_x - min_x) / (width - 50) * 25,
             (max_y - min_y) / (height - 50) * 25) + 0.1

min_x -= buffer
max_x += buffer
min_y -= buffer
max_y += buffer

# icon_str = \
#     ["                                                  ",
#      "               ##############                     ",
#      "             ###################                  ",
#      "            #####################                 ",
#      "           ###            ##    ##                ",
#      "           ##             ##     ##               ",
#      "          ###             ##     ##               ",
#      "          ###             ##      ##              ",
#      "          ###             ##      ##              ",
#      "          ###             ##       ##             ",
#      "          ###             ##       ##             ",
#      "          ###             ##        ##            ",
#      "          ####            ##        ##            ",
#      "         ##############################           ",
#      "      ####################################        ",
#      "    ########################################      ",
#      "   ###########################################    ",
#      "  ############################################### ",
#      "  ############################################### ",
#      " #################################################",
#      " #######     #######################     #########",
#      "######   ###   ###################   ###   #######",
#      "#####  #######  #################  #######  ######",
#      "#####  #######  #################  #######  ######",
#      "####  #########  ###############  #########  #####",
#      " ###  #########  ###############  #########  ###  ",
#      "      #########                   #########       ",
#      "       #######                     #######        ",
#      "       #######                     #######        ",
#      "         ###                         ###          "]

icon_str = \
    ["         ######               ",
     "        #########             ",
     "       ###     ###            ",
     "       ##       ###           ",
     "       ##        ##           ",
     "       ##        ##           ",
     "       ##        ##           ",
     "       ##        ###          ",
     "     ####        #####        ",
     "  ########################    ",
     " ###########################  ",
     " ############################ ",
     " ############################ ",
     "####      #########      #####",
     "###   ##   #######   ##   ####",
     "##   ####   #####   ####   ###",
     "##  ######  #####  ######  ###",
     "    ######         ######     ",
     "     ####           ####      ",
     "      ##             ##       ",
     ]

cpo_color = np.array([216, 83, 201], dtype=np.uint8)
revel_color = np.array([46, 92, 175], dtype=np.uint8)
static_color = np.array([151, 156, 9], dtype=np.uint8)

iwidth = len(icon_str[0])
iheight = len(icon_str)
cpo_icon = np.full((iwidth, iheight, 3), 255, dtype=np.uint8)
revel_icon = np.full((iwidth, iheight, 3), 255, dtype=np.uint8)
static_icon = np.full((iwidth, iheight, 3), 255, dtype=np.uint8)

for i in range(iheight):
    for j in range(iwidth):
        if icon_str[i][j] == '#':
            cpo_icon[j, i] = cpo_color
            revel_icon[j, i] = revel_color
            static_icon[j, i] = static_color

images = np.full((frames, width, height, 3), 255, dtype=np.uint8)
obs_min_x = int(width * (1. - min_x) / (max_x - min_x))
obs_max_x = int(width * (2. - min_x) / (max_x - min_x))
obs_min_y = int(height * (1. - min_y) / (max_y - min_y))
obs_max_y = int(height * (2. - min_y) / (max_y - min_y))
goal_min_x = int(width * (3. - min_x) / (max_x - min_x))
goal_min_y = int(height * (3. - min_y) / (max_y - min_y))

for j in range(frames):
    i = dilation * j
    if i < cpo_traj.shape[0]:
        cpo_x = int(width * (cpo_traj[i, 0] - min_x) / (max_x - min_x))
        cpo_y = int(height * (cpo_traj[i, 1] - min_y) / (max_y - min_y))
    else:
        cpo_x = int(width * (cpo_traj[-1, 0] - min_x) / (max_x - min_x))
        cpo_y = int(height * (cpo_traj[-1, 1] - min_y) / (max_y - min_y))
    if i < revel_traj.shape[0]:
        revel_x = int(width * (revel_traj[i, 0] - min_x) /
                      (max_x - min_x))
        revel_y = int(height * (revel_traj[i, 1] - min_y) /
                      (max_y - min_y))
    else:
        revel_x = int(width * (revel_traj[-1, 0] - min_x) /
                      (max_x - min_x))
        revel_y = int(height * (revel_traj[-1, 1] - min_y) /
                      (max_y - min_y))
    if i < static_traj.shape[0]:
        static_x = int(width * (static_traj[i, 0] - min_x) /
                       (max_x - min_x))
        static_y = int(height * (static_traj[i, 1] - min_y) /
                       (max_y - min_y))
    else:
        static_x = int(width * (static_traj[-1, 0] - min_x) /
                       (max_x - min_x))
        static_y = int(height * (static_traj[-1, 1] - min_y) /
                       (max_y - min_y))

    cxl = cpo_x - iwidth // 2
    cxh = cpo_x + (iwidth + 1) // 2
    cyl = cpo_y - iheight // 2
    cyh = cpo_y + (iheight + 1) // 2

    rxl = revel_x - iwidth // 2
    rxh = revel_x + (iwidth + 1) // 2
    ryl = revel_y - iheight // 2
    ryh = revel_y + (iheight + 1) // 2

    sxl = static_x - iwidth // 2
    sxh = static_x + (iwidth + 1) // 2
    syl = static_y - iheight // 2
    syh = static_y + (iheight + 1) // 2

    # print(i)
    # print("Bounds:", min_x, max_x, min_y, max_y)
    # print("Revel:", revel_traj[-1, 0], revel_traj[-1, 1])
    # print("Revel px:", rxl, rxh, ryl, ryh)

    images[j, obs_min_x:obs_max_x, obs_min_y:obs_max_y] = \
        np.array([255, 100, 100], dtype=np.uint8)
    images[j, goal_min_x:, goal_min_y:] = \
        np.array([100, 255, 100], dtype=np.uint8)

    images[j, cxl:cxh, cyl:cyh] = \
        overlay(cpo_icon, images[j, cxl:cxh, cyl:cyh])
    images[j, rxl:rxh, ryl:ryh] = \
        overlay(revel_icon, images[j, rxl:rxh, ryl:ryh])
    images[j, sxl:sxh, syl:syh] = \
        overlay(static_icon, images[j, sxl:sxh, syl:syh])

images = np.transpose(images, axes=(0, 2, 1, 3))

imageio.v3.imwrite('obstacle_comparison.gif', images)
