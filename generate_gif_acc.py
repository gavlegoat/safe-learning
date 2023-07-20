import imageio
from numpy import matrix, array
import numpy as np


if __name__ == "__main__":

    cpo_fn = "trajectories/Traj-ACCEnv-v0234355.txt"
    revel_fn = "trajectory.txt"
    static_fn = "static_trajectory.txt"
    cpo_traj = []
    with open(cpo_fn, 'r') as cpo_file:
        for line in cpo_file:
            cpo_traj.append(eval(line))
    revel_traj = []
    with open(revel_fn, 'r') as revel_file:
        i = 0
        step = ""
        for line in revel_file:
            step += line
            i += 1
            if i == 5:
                i = 0
                tmp = eval(step)
                step = ""
                revel_traj.append(np.squeeze(array(tmp[0][:2])))
    static_traj = []
    with open(static_fn, 'r') as static_file:
        i = 0
        step = ""
        for line in static_file:
            step += line
            i += 1
            if i == 5:
                i = 0
                tmp = eval(step)
                step = ""
                static_traj.append(np.squeeze(array(tmp[0][:2])))

    cpo_next = [cpo_traj[-1][0] + 0.01 * cpo_traj[-1][1], 0]
    cpo_traj.append(cpo_next)
    cpo_traj = array(cpo_traj)
    revel_traj = array(revel_traj)
    static_traj = array(static_traj)

    min_x = min(np.min(cpo_traj[:, 0]),
                np.min(revel_traj[:, 0]),
                np.min(static_traj[:, 0])) - 0.2
    max_x = 0.2

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

    iwidth = len(icon_str[0])
    iheight = len(icon_str)
    icon = np.full((iwidth, iheight), 255, dtype=np.uint8)
    for i in range(iheight):
        for j in range(iwidth):
            if icon_str[i][j] == '#':
                icon[j, i] = 0

    width = 300
    height = 200
    frames = 300
    cpo_y = height // 4
    revel_y = 2 * height // 4
    static_y = 3 * height // 4

    images = np.full((frames, width, height), 255, dtype=np.uint8)
    fixed_x = int(width * (-min_x) / (max_x - min_x))
    fxl = fixed_x
    fxh = fixed_x + iwidth
    for j in range(frames):
        i = 2 * j
        if i < cpo_traj.shape[0]:
            cpo_x = int(width * (cpo_traj[i, 0] - min_x) / (max_x - min_x))
        else:
            cpo_x = int(width * (cpo_traj[-1, 0] - min_x) / (max_x - min_x))
        revel_x = int(width * (revel_traj[i, 0] - min_x) / (max_x - min_x))
        static_x = int(width * (static_traj[i, 0] - min_x) / (max_x - min_x))

        cxl = cpo_x - iwidth
        cxh = cpo_x
        cyl = cpo_y - iheight // 2
        cyh = cpo_y + (iheight + 1) // 2

        rxl = revel_x - iwidth
        rxh = revel_x
        ryl = revel_y - iheight // 2
        ryh = revel_y + (iheight + 1) // 2

        sxl = static_x - iwidth
        sxh = static_x
        syl = static_y - iheight // 2
        syh = static_y + (iheight + 1) // 2

        images[j, cxl:cxh, cyl:cyh] = icon
        images[j, rxl:rxh, ryl:ryh] = icon
        images[j, sxl:sxh, syl:syh] = icon
        images[j, fxl:fxh, cyl:cyh] = icon
        images[j, fxl:fxh, ryl:ryh] = icon
        images[j, fxl:fxh, syl:syh] = icon

    images = np.stack([images] * 3, axis=-1)
    images = np.transpose(images, axes=(0, 2, 1, 3))

    imageio.v3.imwrite('comparison.gif', images)
