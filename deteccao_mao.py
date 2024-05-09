#import the librarys
import mediapipe as mp 
import cv2
import os
from time import sleep
from pynput.keyboard import Controller
from pynput.keyboard import Key

#writing all constants of colors that will be utilize
BRANCO = (255,255,255)
PRETO = (0,0,0)
AZUL = (255,0,0)
VERDE = (0,255,0)
VERMELHO = (0,0,255)
AZUL_CLARO = (255,255,0)
#hands solution to detection
mp_maos = mp.solutions.hands
#draw solution
mp_desenho = mp.solutions.drawing_utils

#machine learning model capable of recognizing hands in the image
maos = mp_maos.Hands()

#connecting with webcam using function VideoCapture
camera = cv2.VideoCapture(0)

#increasing the resolution image
#resolution image
resolucao_x = 1280
resolucao_y = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao_x)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao_y)
#creating a var to support that inform if the file is open or closed (and not open the file in loop)
bloco_notas = False
chrome = False
spotify = False
#creating a list with all letters 
teclas = [['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A','S','D','F','G','H','J','K','L'],
            ['Z','X','C','V','B','N','M', ',','.',' ']]

offset = 50
contador = 0
texto = '>'

#creating keyboard
teclado = Controller()


#creating a function to extract the information of reference points coordinates of hands
def encontra_coordenadas_maos(img, lado_invertido = False):
    #transforming the images in RGB to the mediapipe to process
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # -> (img to process, type color)
    
    #processing the image with machine learning model
    resultado = maos.process(img_rgb) # -> returns: multi_hand_landmarks, multi_hand_word_landmarks, multi_handedness
    
    todas_maos = []
    
    #if haven't hands on screen the webcam not to capture
    if resultado.multi_hand_landmarks:
        #collecting information from side of the hand inside the function
        for lado_mao, marcacao_maos in zip(resultado.multi_handedness, resultado.multi_hand_landmarks):
            '''storage the coordinates in vars to use in this project, through multi_hand_landmarks
                identify the pixel position of this reference point'''
                
            #creating a dictionary to storage this coordinates information
            info_mao = {}
            #creating a list to storage the coordinates
            coordenadas = []
            
            #print(marcacao_maos) -> see how the API returns the coordinates
            for marcacao in marcacao_maos.landmark: #-> accessing each landmark
                #transforming each coordinate in pixel(int), multiplying the coordinate by resolution
                coord_x, coord_y, coord_z = int(marcacao.x * resolucao_x), int(marcacao.y * resolucao_y), int(marcacao.z * resolucao_x)
                coordenadas.append((coord_x, coord_y, coord_z))
            
            #creating a dictionary
            info_mao['coordenadas'] = coordenadas
            
            #creating a logic to reverse left and right
            if lado_invertido:
                if lado_mao.classification[0].label == 'Left':
                    info_mao['lado'] = 'Right'
                else:
                    info_mao['lado'] = 'Left'
                    
            else:
                info_mao['lado'] = lado_mao.classification[0].label
            
            
            #storing in dictionary
            todas_maos.append(info_mao)
            
            #drawing points in img, the coordinates pointers and the hand connections
            mp_desenho.draw_landmarks(img,
                                    marcacao_maos,
                                    mp_maos.HAND_CONNECTIONS)
            
    return img,todas_maos


#function to find out how many fingers on the hand are raised
def dedos_levantados(mao):
    '''to extract what fingers are raised\n
        inform the dictionary as a parameter of this function'''
        
    #creating a list to storage the info if the finger are raised or no, putting as true or false
    dedos = []
    #comparing wheter the fingertip has a coorinate x lower than the mean of coordinate position of finger
    if mao['lado'] == 'Right': 
        if mao['coordenadas'][4][0] < mao['coordenadas'][3][0]:
            dedos.append(True)
        else:
            dedos.append(False)
    else:
        if mao['coordenadas'][4][0] > mao['coordenadas'][3][0]:
            dedos.append(True)
        else:
            dedos.append(False)
    
    #using a for loop, to cycle through each of the fingertip index values ​​which are 4,8, 12, 16 and 20
    for ponta_dedo in [8,12,16,20]:
        #comparing whether the fingertip has a coordinate y lower than the mean of coordinate position of finger
        if mao['coordenadas'][ponta_dedo][1] < mao['coordenadas'][ponta_dedo-2][1]:
            #check if the finger is raising (true: finger raised; false: finger down)
            dedos.append(True)
        else:
            dedos.append(False)
            
    return dedos

#define the function for creating keys
def imprime_botoes(img, posicao, letra, tamanho = 50, cor_retangulo = BRANCO):
    """show buttons

    Args:
        img (array): img captured by webcam
        posicao (list): coord x,y,z
        letra (str): letter of keyboard
        tamanho (int, optional): size of rectangle. Defaults to 50.
        cor_retangulo (tuple, optional): color of rectangle. Defaults to BRANCO.
    """
    #using function rectangle to do draw in keyboard
    cv2.rectangle(img, posicao, (posicao[0]+ tamanho, posicao[1]+tamanho),cor_retangulo, cv2.FILLED)
    cv2.rectangle(img, posicao, (posicao[0]+ tamanho, posicao[1]+tamanho),AZUL, 1)
    #utilizing a function putText to writ the text on keyboard
    cv2.putText(img, letra, (posicao[0]+15, posicao[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, PRETO, 2)
    return img

#showing the image on screen using a loop while
while True:
    
    #reading the images by webcam, using function read() that return 2 values
    sucesso, img = camera.read() #returns the image in BRG
    
    #inverting the image
    img = cv2.flip(img, 1)
    
    #calling function
    img,todas_maos = encontra_coordenadas_maos(img)
    
    #checking if i only have one hand raised
    if len(todas_maos) == 1:
        info_dedos_mao1 = dedos_levantados(todas_maos[0])
        #adding conditional to appear the keyboard if the right is raised
        if todas_maos[0]['lado'] == 'Left':
            #to extract the coord of index finger
            indicador_x, indicador_y, indicador_z =  todas_maos[0]['coordenadas'][8]
            
            #to show coordinate z
            cv2.putText(img, f'Distancia camera: {indicador_z}', (850, 50), cv2.FONT_HERSHEY_COMPLEX, 1, BRANCO, 2)
            for indice_linha, linha_teclado in enumerate(teclas):
                for indice, letra in enumerate(linha_teclado):
                    if sum(info_dedos_mao1) <= 1:
                        letra = letra.lower()
                    img = imprime_botoes(img, (offset+indice*80, offset + indice_linha*80), letra)
                    
                    #to see if the coordinates of index finger are positioned in keyboard region
                    if offset + indice*80 < indicador_x < 100+indice * 80 and offset + indice_linha*80 < indicador_y< 100 +indice_linha*80:
                        #add green color the selected key
                        img = img = imprime_botoes(img, (offset+indice*80, offset + indice_linha*80), letra, cor_retangulo=VERDE)
                        
                        if indicador_z < - 85:
                            #to create a count as a way to digitalize the keyboard
                            contador = 1
                            escreve = letra
                            img = img = imprime_botoes(img, (offset+indice*80, offset + indice_linha*80), letra, cor_retangulo=AZUL_CLARO)
            if contador:
                contador+=1
                if contador == 3:
                    texto += escreve
                    contador = 0
                    teclado.press(escreve)
                    
            #to check if the little finger is raised and the text has at least one character
            if info_dedos_mao1 == [False, False, False, False, True] and len(texto) > 1:
                texto = texto[:-1]
                teclado.press(Key.backspace)
                sleep(0.15)
            
            #to create a session where will be show the text that we are typing using a function rectangle
            cv2.rectangle(img, (offset, 450), (830, 500), BRANCO, cv2.FILLED)
            cv2.rectangle(img, (offset, 450), (830, 500), AZUL, 1)
            
            #to put the text with a function putText
            cv2.putText(img, texto[-40:], (offset, 480), cv2.FONT_HERSHEY_COMPLEX, 1, PRETO, 2)
            #to add a circle in the finger tip
            cv2.circle(img, (indicador_x, indicador_y), 7, AZUL, cv2.FILLED)
            
        #to open the files the left hand should raised
        if todas_maos[0]['lado'] == 'Right':
            #if the index finger is raised the code will open the block note
            if info_dedos_mao1 == [False, True, False, False, False] and bloco_notas == False:
                bloco_notas = True
                os.startfile(r'C:\Windows\system32\notepad.exe')
            
            #if the index finger is raised the code will open the chrome
            if info_dedos_mao1 == [False, True,  True, False, False] and chrome == False:
                chrome = True
                os.startfile(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
                
            #if the index finger is raised the code will open the spotify
            if info_dedos_mao1 == [False, True, True, True, False] and spotify == False:
                spotify = True
                os.startfile(r"C:\Users\joaov\AppData\Roaming\Spotify\Spotify.exe")
                
            #closing the notepad
            if info_dedos_mao1 == [False, False,False,False, False] and bloco_notas == True:
                bloco_notas = False
                os.system('TASKKILL  /IM notepad.exe')
            
            #closing the chrome    
            if info_dedos_mao1 == [False, True, True, True, True] and chrome == True:
                chrome = False
                os.system('TASKKILL /IM chrome.exe')
                
            #closing the spotify
            if info_dedos_mao1 == [True, False, False, False, True] and spotify == True:
                spotify = False
                os.system('TASKKILL /IM Spotify.exe')
                
            #closing the program
            if info_dedos_mao1 == [False, True, False, False, True]:
                break

    #showing image, inputting the name of the screen and input the image
    cv2.imshow('Imagem', img)
    
    #taking the frame of image
    tecla = cv2.waitKey(1)
    
    if tecla == 27:
        break
    
#saving in a file the text typed
with open('text.txt', 'w') as arquivo:
    arquivo.write(texto)




    