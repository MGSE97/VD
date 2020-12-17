import os
import glob

def create_video(save_dir):
    imageNames = "images.txt"
    fileName = "sph.mp4"
    fps = 10
    duration = 1/fps

    if not os.path.exists(save_dir+imageNames):
        image_files = list(sorted(filter(os.path.isfile, glob.glob("{}*.png".format(save_dir))), key=lambda f: int(os.path.basename(f)[3:-4])))
        with open(save_dir+imageNames, "w") as f:
            f.write("\n".join(["file '{}'\nduration {}".format(os.path.basename(f), duration) for f in image_files]))

    command = "cd {} && ffmpeg -r {} -c:v libx264 -crf 0 -c:a aac -strict -2 -y {} -f concat -i {}".format(save_dir, fps, fileName, imageNames)
    print(command)
    os.chdir(save_dir)
    os.system(command)
    os.system('"{}"'.format(fileName))
    os.chdir("..")

create_video("D:/Documents/Projekty/Škola/VD/cv5/sph_17_40x40x40_mu_3_5_iso_81_v_3/")