# Detecção de Objetos com diferentes pré-processamentos

Para isntalar as dependências do projeto e executá-lo, utilize os seguintes comandos:

```sh
# aqui ja ira instalar as dependencias do arquivo requirements.txt
./install.sh
```

O script install.sh irá instalar as dependências dentro de um ambiente virtual, então é necessário ativá-lo antes de executar o código:

```sh
# ativa ambiente virtual
. .env/bin/activate
```

Para realizar a detecção é necessário alguns argumentos:
* --imagem: caminho da imagem de entrada
* --prototxt: caminho para Caffe 'deploy' prototxt file
* --model: caminho para o modelo Caffe pre-treinado
* --confidence (opcional): probabilidade minima para filtrar deteccoes fracas

Exemplo de execução:

```sh
python object-detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel --imagem imagens/example_01.jpg
```

