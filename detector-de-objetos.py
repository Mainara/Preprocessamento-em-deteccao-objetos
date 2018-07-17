# pacote utilizado para auxiliar na manipulacao de listas
import numpy as np

# pacote utilizado para auxiliar a interface de comando de linha
import argparse

# pacote utilizado para auxiliar a manipulacao de imagemns
import cv2


import os
 
class DetectorObjetos:

    args = None

    # construcao dos argumentos necessarios para executar a deteccao
    def getArgs(self):
        global args
        argp = argparse.ArgumentParser()
        argp.add_argument("-i", "--imagens", required=True, help="caminho para as imagens de entrada")
        argp.add_argument("-p", "--prototxt", required=True, help="caminho para Caffe 'deploy' prototxt file")
        argp.add_argument("-m", "--model", required=True, help="caminho para o modelo Caffe pre-treinado")
        argp.add_argument("-c", "--confidence", type=float, default=0.2, help="probabilidade minima para filtrar deteccoes fracas")
        args = vars(argp.parse_args())


    # carraga o arquivo do modelo
    def carregaModelo(self):
        global args
        print("[INFO] carregando modelo...")
        net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
        self.carregaImagens(net)

    # carrega a imagem de entrada e constroi um blob de entrada para a imagem
    # redimencionando a mesma para 300x300 pixels e fazendo a normalizacao
    # (info: a normalizacao eh feita pelo as autores que implementaram o MobileNet SSD)
    def carregaImagens(self, net):
        global args
        caminho_imagens = args["imagens"]
        imagens = os.listdir(caminho_imagens)
        for img in imagens:
            imagem = cv2.imread(caminho_imagens +img)
            (h, w) = imagem.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(imagem, (300, 300)), 0.007843, (300, 300), 127.5)
            self.getDeteccoes(net, blob, w, h, imagem)
        

    # passa o blob pela redes neurais e obtem as deteccoes e predicoes
    def getDeteccoes(self, net, blob, w, h, imagem):
        print("[INFO] calculando as deteccoes...")
        net.setInput(blob)
        deteccoes = net.forward()
        self.loopSobreDeteccoes(deteccoes, imagem, w, h)
        

    # loop sobre as detccoes
    def loopSobreDeteccoes(self, deteccoes, imagem, w, h):

        # inicializa os labels que o MobileNet SSD foi treinado para
        # detectar, e entao gerar os bounding boxes de CORES para cada classe
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	    "sofa", "train", "tvmonitor"]
        CORES = np.random.uniform(0, 255, size=(len(CLASSES), 3))

        for i in np.arange(0, deteccoes.shape[2]):
	        # extrai a probabilidade associada com a predicao
	        confidence = deteccoes[0, 0, i, 2]
 
            # filtra as deteccoes fracas de acordo com o calor de `confidence`
	        if confidence > args["confidence"]:
		        # extrai o indice da classe com determinado label em `deteccoes`,
		        # e calcula as coordenadas (x, y) do bound box criado para o objeto
		        idx = int(deteccoes[0, 0, i, 1])
		        box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
		        (comecoX, comecoY, fimX, fimY) = box.astype("int")
		        
                # exibe a previsao
		        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
		        print("[INFO] {}".format(label))
		        cv2.rectangle(imagem, (comecoX, comecoY), (fimX, fimY),
			        CORES[idx], 2)
		        y = comecoY - 15 if comecoY - 15 > 15 else comecoY + 15
		        cv2.putText(imagem, label, (comecoX, y),
			        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORES[idx], 2)
        self.mostraImagem(imagem)

    # mostra a imagem de saida
    def mostraImagem(self, imagem):
        cv2.imshow("Output", imagem)
        cv2.waitKey(0)

if __name__ == "__main__":
    od = DetectorObjetos()
    od.getArgs()
    od.carregaModelo()
    
