# Pipeline de Anotacao 360

Esse repositorio é a entrega do desafio tecnico de anotacao automatizada em imagens 360.

O problema principal aqui é que imagens equiretangulares distorcem, entao rodar um modelo de IA direto nelas nao é o ideal. A solucao implementada foi extrair varias perspectivas (simulando cameras virtuais), passar o GroundingDINO nelas pra detectar os carros, e depois fazer a matematica reversa pra desenhar os bounding boxes de volta na imagem original.

## Como rodar
Fiz o setup usando o `pyproject.toml` pra nao ter erro de versao com as dependencias.

1. Criar um ambiente virtual sua maquina:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Instalar as dependencias (via .toml)
```bash
pip install .
```

3. rodar codigo principal
```bash
python3 main.py --pitch {valor} --yaw {valor} --fov {valor}
```
valores default se nao forem passados como parametros:
    --pitch:30
    --yaw:45
    --fov:90

## Sobre a Reprojeção:
Na foto plana, o bounding box é um quadrado perfeito. Quando a gente projeta isso de volta pro mapa 360, o poligono entorta e as linhas ficam curvas. Resolvi isso da seguinte maneira:

Fiz a parte da reprojecao do zero pois nao achei uma lib muito adequada pra isso. Algumas libs ate se assemelham (e.g. astropy_spherical_to_cartesian), mas nao lidavam muito bem com o que foi proposto.

A ideia foi criar uma funcao que faz a projecao reversa de um pixel especifico da perspectiva plana de volta para a matriz equirectangular.

Dessa maneira, eu jogo os 4 cantos do bounding box local nessa funcao pra achar o (x,y) de cada vertice na imagem 360 global. Depois eu tiro o min() e o max() desses 4 pontos distorcidos.

Com isso, eu garanto a criacao de um novo retangulo reto e alinhado com os eixos da imagem que engloba 100% da area do objeto, sem cortar nenhum pedaço do carro.

## Sobreposicao e NMS
Como a gente faz uma varredura girando a camera, o mesmo carro acaba aparecendo em mais de uma vista e gera deteccoes duplicadas.

Pra limpar isso, juntei as coordenadas finais e a taxa de confianca (score) de todas as caixas numa lista so e apliquei o algoritmo de NMS (cv2.dnn.NMSBoxes) do OpenCV. Ele calcula a intersecao e deixa so um bounding box definitiva por carro


## Desafios:
A parte mais complexa foi conseguir de fato entender e aprender o que foi proposto. Nao conhecia muitos conceitos e nem havia trabalhado com deteccao neste sentido.

Incrivelmente achei a parte matematica mais intuitiva, e, como eu ja havia mexido com matrizes de rotacao e afins, consegui entender as projecoes de forma mais facil.

Achei o conceito de Zero-Shot complexo e nao muito simples de compreender. Achei a documentacao do Audistill/GroundingDINO escassa e sem sentido.
    
    
## Comentarios:

A deteccao ainda nao esta perfeita e o codigo pode ser em muito melhorado.
Mesmo ajustando alguns parametros, a deteccao falha em alguns pontos.
      
Adicionei um documento "sources.txt" do qual cita as principais referencias que usei.
    
Usei IA de forma muito restrita, e nunca para me dar codigo pronto. Ela me auxiliou com a estruturacao, quando eu me perdia e nao sabia os proximos passos, e tambem com alguns pontos de debug. Ex: Me ajudou a debugar uma imcompatibilidade de biblioteca antiga com o numpy (da qual pode ser vista no codigo principal com comentario --PATCH--)
   
   
## Resultados

### Exemplo

Antes do processamento:
![Original 1](foto_original.jpg)

Depois do pipeline:
![Resultado 1](foto_detected.jpg)


## Video Explicativo

Fiz um video explicando as etapas do codigo. Esta no youtube:

URL: ""
