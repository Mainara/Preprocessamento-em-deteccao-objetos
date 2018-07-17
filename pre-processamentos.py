# pacote utilizado para auxiliar a manipulacao de imagemns
import cv2

import numpy as np

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import morphology

from skimage.measure import label
from skimage.morphology import closing
from skimage.feature import canny


class Preprocessamentos:

    def dilatacao(self, img, tipo_kernel):
        kernel = None
        if tipo_kernel == "rect":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        elif tipo_kernel == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        elif tipo_kernel == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        imagem_dilatada = cv2.dilate(img, kernel, iterations=1)
        return imagem_dilatada

    def erosao(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(img, kernel, iterations=1)

    # abertura: erosao + dilatacao
    def abertura(self, img):
        selem = disk(6)
        return opening(img, selem)

    # fechamento: dilatacao + erosao
    def fechamento(self, img):
        selem = disk(6)
        return closing(img, selem)

    def sequencialAbertura_Fechamento(self, img):
        img_opened = self.abertura(img)
        return self.fechamento(img_opened)

    def sequencialFechamento_Abertura(self, img):
        img_closed = self.fechamento(img)
        return self.abertura(img_closed)

    def segmentacaoRegioes(self, img):
        distance = ndimage.distance_transform_edt(img)
        local_maxi = peak_local_max(
            distance, indices=False, footprint=np.ones((3, 3)), labels=img)
        markers = morphology.label(local_maxi)
        labels_ws = watershed(-distance, markers, mask=img)
        return labels_ws

    def deteccaoBordas(self, img):
        edges = cv2.Canny(img, 100, 200)

        return edges

    
if __name__ == "__main__":
    pp = Preprocessamentos()
    img = cv2.imread("imagens/chair/000005.jpg", 0)
    cv2.imshow("Original", img)

    '''
    img_dilatada = pp.dilatacao(img, "cross")
    cv2.imshow("Dilatacao", img_dilatada)
    cv2.imwrite("dilatacao.jpg", img_dilatada)


    img_eroded = pp.erosao(img) 
    cv2.imshow("Erosao", img_eroded)
    cv2.imwrite("erosao.jpg", img_eroded)

    
    img_opened = pp.abertura(img)
    cv2.imshow("Abertura", img_opened)
    cv2.imwrite("abertura.jpg", img_opened)
    
    img_closed = pp.fechamento(img)
    cv2.imshow("Fechamento", img_closed)
    cv2.imwrite("fechamento.jpg", img_closed)
    cv2.waitKey(0)
    '''
    cv2.imshow("kjb", pp.deteccaoBordas(img))
    cv2.waitKey(0)
