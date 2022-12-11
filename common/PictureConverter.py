# 画像の変形
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import cv2

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


def PictureConvert():
    """
    画像変換処理
    """
    thumb_width = 200
    # １．Webカメラで手書き文字を取得(640×480)
    im = Image.open("mnist_data\\Number.jpg").convert('L')
    plt.imshow(im)

    # ２．正方形(200×200)で切り出し
    im_thumb = crop_max_square(im).resize((thumb_width, thumb_width), Image.LANCZOS)
    im_thumb.save("mnist_data\\astronaut_thumbnail_max_square.jpg", quality=95)

    # ３．グレースケール・白黒を反転
    img2 = cv2.imread("mnist_data\\astronaut_thumbnail_max_square.jpg")
    im_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    #白黒反転作業作業
    im_gray = 255 - im_gray
    cv2.imwrite('mnist_data\\Number_gray.jpg', im_gray)

    # ４．2値価
    th, im_gray_th_otsu = cv2.threshold(im_gray, 180, 255, cv2.THRESH_OTSU)
    cv2.imwrite('mnist_data\\Number_wb.jpg', im_gray_th_otsu)