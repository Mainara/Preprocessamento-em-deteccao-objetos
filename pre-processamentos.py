# pacote utilizado para auxiliar a manipulacao de imagemns
import cv2
import numpy as np
import os
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import disk
from scipy import ndimage, misc, signal
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import morphology
from skimage.measure import label
from skimage.morphology import closing
from skimage.feature import canny


class Preprocessamentos:

    # mehora imagem

    def agucamentoBordas(self, caminho):
        img = misc.imread(caminho)/255.
        filtro_agucamento = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        img_com_filtro = np.ones(img.shape)
        classe = caminho.split("/")[1]
        nome_img = caminho.split("/")[2]
        for i in range(3):
            img_com_filtro[...,i] = np.clip(signal.convolve2d(img[...,i], filtro_agucamento, mode='same', boundary='symm'), 0,1)
        if not os.path.exists("preprocessamento/agucamento/" + classe ):
			os.makedirs("preprocessamento/agucamento/" + classe)
        misc.imsave("preprocessamento/agucamento/"+ classe + "/" + nome_img , img_com_filtro)
    
    def equalizaHist(self, caminho):
        img = cv2.imread(caminho, 0)
        equ = cv2.equalizeHist(img)
        classe = caminho.split("/")[1]
        nome_img = caminho.split("/")[2]
        if not os.path.exists("preprocessamento/hist_equ/"+classe):
			os.makedirs("preprocessamento/hist_equ/"+classe)
        cv2.imwrite("preprocessamento/hist_equ/"+ classe + "/" + nome_img, equ)

    def equalizaHistAdaptativo(self, caminho):
        img = cv2.imread(caminho, 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(img)
        classe = caminho.split("/")[1]
        nome_img = caminho.split("/")[2]
        if not os.path.exists("preprocessamento/hist_equ_adp/" + classe):
			os.makedirs("preprocessamento/hist_equ_adp/" + classe)
        cv2.imwrite("preprocessamento/hist_equ_adp/"+ classe + "/" + nome_img, cl1)

    # piora imagem
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


    def getHistogram(imgIn):
        nValues = 10
        nBins = 10
        bRange = nValues/nBins
        axis=1
        hist = np.zeros((axis, nBins))
        for row in range(0, len(imgIn)):	
            for col in range(0, len(imgIn[row])):
                index = int(imgIn[row,col]/bRange)
                hist[0][index] = hist[0][index] + 1
        return hist


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
    imagens = os.listdir("imagens")
    for classe in imagens:
        caminho_classe = "imagens/"+classe
        imgs = os.listdir(caminho_classe)
        for img in imgs:
            caminho_img = caminho_classe + "/" + img
            print caminho_img
            pp.agucamentoBordas(caminho_img)
            pp.equalizaHist(caminho_img)
            pp.equalizaHistAdaptativo(caminho_img)
    