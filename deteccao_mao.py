#import the librarys
import mediapipe as mp 
import cv2

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

#creating a function to extract the information of reference points coordinates of hands
def encontra_coordenadas_maos(img):
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
            
            #storing in dictionary
            info_mao['coordenadas'] = coordenadas
            todas_maos.append(info_mao)
            
            #drawing points in img, the coordinates pointers and the hand connections
            mp_desenho.draw_landmarks(img,
                                    marcacao_maos,
                                    mp_maos.HAND_CONNECTIONS)
            
    return img,todas_maos

#showing the image on screen using a loop while
while True:
    
    #reading the images by webcam, using function read() that return 2 values
    sucesso, img = camera.read() #returns the image in BRG
    
    #inverting the image
    img = cv2.flit(img, 1)
    
    #calling function
    img,todas_maos = encontra_coordenadas_maos(img)

    #showing image, inputting the name of the screen and input the image
    cv2.imshow('Imagem', img)
    
    #taking the frame of image
    tecla = cv2.waitKey(1)
    
    if tecla == 27:
        break
    
    

#return the information from side of the hand