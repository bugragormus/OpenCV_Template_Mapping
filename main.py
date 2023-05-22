import cv2
import numpy as np
import glob
from itertools import combinations


def find_similar_images(images, template_size):
    similarities = []

    # Döngüdeki tüm görsel kombinasyonlarını al
    for img1_index, img2_index in combinations(range(len(images)), 2):
        img1 = images[img1_index]
        img2 = images[img2_index]

        # Resimleri gri tonlamalı olarak yükle
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Şablon boyutuna göre yeniden boyutlandırma yap
        resized1 = cv2.resize(gray1, template_size)
        resized2 = cv2.resize(gray2, template_size)

        # Template matching işlemi
        result = cv2.matchTemplate(resized1, resized2, cv2.TM_CCOEFF_NORMED)
        _, similarity, _, _ = cv2.minMaxLoc(result)

        similarities.append((img1_index, img2_index, similarity))

    # Benzerlik yüzdelerine göre sırala
    sorted_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)

    return sorted_similarities


# Ana program
image_directory = "/ALGORTIMA_ANALIZI_ODEV_SOURCES/*.png"  # Görsellerin_bulunduğu_klasör/*.dosya_uzantısı
template_size = (20, 20)  # Şablon boyutu (genişlik, yükseklik)

images = []
image_paths = glob.glob(image_directory)  # Klasördeki tüm görsel dosya yollarını al

for path in image_paths:
    img = cv2.imread(path)
    images.append(img)

similar_images = find_similar_images(images, template_size)

for img1_index, img2_index, similarity in similar_images:
    img_path1 = image_paths[img1_index]
    img_path2 = image_paths[img2_index]
    print(f"Benzerlik Oranı: {similarity:.2f} -> {img_path1} ve {img_path2}")
