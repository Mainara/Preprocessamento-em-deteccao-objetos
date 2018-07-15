# pacote utilizado para auxiliar a manipulacao de imagemns
import cv2

class Preprocessamentos:

    def dilatacao(self, img, tipo_kernel):
        kernel = None
        if tipo_kernel == "rect":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        elif tipo_kernel == "ellipse":
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        elif tipo_kernel == "cross":
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        imagem_dilatada = cv2.dilate(img,kernel,iterations = 1)
        return imagem_dilatada

if __name__ == "__main__":
    pp = Preprocessamentos()
    img = cv2.imread("imagens/example_01.jpg", 0)
    cv2.imshow("output1", img)
    imgsaida = pp.dilatacao(img, "cross")
    cv2.imshow("output", imgsaida)
    cv2.imwrite("img.jpg", imgsaida)
    cv2.waitKey(0)