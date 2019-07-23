from PIL import Image
import os.path
import glob


def convertjpg(jpgfile, outdir, width=100, height=100):
    """
    # jpg图片更改尺寸后保存到outdir目录下
    :param jpgfile:
    :param outdir: 输出目录
    :param width:输出图片宽度
    :param height:输出图片高度
    :return:None
    """
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.ANTIALIAS)
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


if __name__ == "__main__":
    path = 'E:/ProgrammeCode/IDEA/PV_Panel_Classify'
    for jpg_file in glob.glob(path+"/0510/N/*.jpg"):
        convertjpg(jpg_file, path+"/0510_resize/N")
    for jpg_file in glob.glob(path+"/0510/P/*.jpg"):
        convertjpg(jpg_file, path+"/0510_resize/P")
