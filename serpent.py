#ctrl K 0 pour collapse all // Ctrl K J pour uncollapse

import pygame
import time
import random
import numpy as np
pygame.init()
import tensorflow as tf
from tensorflow import keras
from collections import deque
import os
import sys
import copy
from keras import backend as K
import concurrent.futures
from keras.optimizers import SGD
from tensorflow.python.client import device_lib


random.seed(2)
np.random.seed(2)

# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # -1 pour desactiver GPU, 0 sinon
# print(tf.__version__)
print('A: ', tf.test.is_built_with_cuda)

tf.compat.v1.disable_eager_execution() #pour accelerer
# tf.compat.v1.disable_v2_behavior()

nbrgen_compteur = 0
nbrindividu = 0 

largeur, hauteur = 260, 260
taille_case = 20
vitesse = 100000000000000000000000

blanc = (255, 255, 255)
rouge = (255, 0, 0)
vert = (30 , 249 , 158)
vertdif = (0, 150, 0)
noir = (0, 0, 0)

ecran = pygame.display.set_mode((largeur,hauteur))
pygame.display.set_caption("Snake Game")

font = pygame.font.Font(None, 20)  

def afficher_texte(texte, x, y, couleur):
    texte_surface = font.render(texte, True, couleur)
    ecran.blit(texte_surface, (x, y))

taille=20

def create_nn():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=120, input_dim=taille, activation='relu'))
    # model.add(keras.layers.Dense(units=60, input_dim=taille, activation='relu'))
    # model.add(keras.layers.Dense(units=16, input_dim=taille, activation='relu'))      
    # model.add(keras.layers.Dense(units=20, input_dim=taille, activation='sigmoid', bias_initializer='RandomNormal')) # , bias_initializer='RandomNormal'
    # model.add(keras.layers.Dense(units=120, activation='relu'))
    # model.add(keras.layers.Dense(units=120, activation='relu')) 
    model.add(keras.layers.Dense(units=4, activation='sigmoid'))
    return model

model=create_nn()
model.compile(optimizer='adam', loss='mse')
nomfichier = 'best.h5'
# if os.path.exists(nomfichier):
#     model = keras.models.load_model(nomfichier)



def predictmaxi_nn(state):
    
    q_values = model.predict(state, verbose=0)[0]
    best_action = inverser2(np.argmax(q_values))
    return best_action, np.max(q_values)


def jeu(modescore=0, bool_random=0, mode="ff"):
    if modescore==1:
        bool_random= 0

    Nsave=2000000000000
    Nmin=200
    N=Nsave
    pp=0
    p=0
    
    if mode=='nntrain':
        x_train =np.empty((N,taille))
        y_train =np.empty((N,4))

    while True:
        initial_position = [random.randint(0, largeur // taille_case - 1) * taille_case,
                        random.randint(0, hauteur // taille_case - 1) * taille_case]
        # ppp= np.random.choice([-1, 1])
        # pose1 =
        # pose2 = 
        serpent = [initial_position]
        direction = 'DROITE'
        fruit = generer_fruit(serpent)
        collision=0
        newdir=direction
        n=4
        temps_reinit=0
        temps2=0
        no_change_dir = 1
        pommes=0
        stop=0
        non_mort = 0
        scoregit=0
        penalite_same_dir=0
        avg_steps=0
        count_same_direction=0
        cmpt=0
        first_dir= 0
        compteur_repetition=0
        repet0=[-1,-1]
        repet1=[-1,-1]
        repet2=[-1,-1]
        repet3=[-1,-1]
        is_repet1=0
        is_repet2=0
        is_repet3=0
        while collision==0:
            # t1=time.time()

            if mode == 'maxi':
                n=n if len(serpent)<=35 else n+0
                
                newdir = predictmaxi(serpent,fruit, direction, n)[0]
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()

            elif mode == 'nn' or mode=='nntrain':

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                serpenttotal = []
                
                serpenttotal += [if_fruit(serpent,fruit, "GAUCHE"), if_fruit(serpent,fruit, "DROITE"), if_fruit(serpent,fruit, "HAUT"), if_fruit(serpent,fruit, "BAS")]
                
                vecteurdirection = [0]*4
                vecteurdirection[inverser(direction)]=1
                vecteur_distance_mur = distance_mur(serpent)
                vecteur_distance_queue = distance_queue(serpent)
                vecteurobstacle = presence_obstacle(serpent)
                vecteurobstaclequeue = presence_queue(serpent)
                vecteurobstaclequeueloin = presence_queue_loin(serpent)
                vecteurobstaclemur = presence_mur(serpent)
                vecteurfruit = direction_fruit_vecteur(serpent, fruit)
                state_ameliore = np.array(vecteur_distance_mur + vecteurobstaclequeueloin + vecteurobstaclequeue + vecteurfruit).reshape(1,taille)
                new_serpent= serpent[:]
                if np.random.rand() > bool_random:
                    # t3 = time.time()
                    newdir = predictmaxi_nn(state_ameliore)[0]
                    # t4 = time.time()
                else:
                    newdir =np.random.choice(['HAUT', 'BAS', 'GAUCHE', 'DROITE'])
                if temps2==0:
                    first_dir=newdir
                deplacer_serpent(new_serpent, newdir)
                if mode=='nntrain':
                    x_train[p,:] = state_ameliore
                # y_train[p,:] = array_transformed
                p+=1
                
                # reward = recompense(serpent, new_serpent, fruit)  
                if p==N and mode=='nntrain':
                    p=0
                    model.fit(x_train, y_train,epochs=20,batch_size=32)                    
                    model.save(nomfichier)
                    pp+=1
                    x_train =np.empty((N,taille))
                    y_train =np.empty((N,4))
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.KEYDOWN:
                        newdir = invmanuel(event.key)
                
                print(presence_queue_loin(serpent))
            
            old_direction = direction

            if newdir == 'GAUCHE' and direction != 'DROITE':
                direction = 'GAUCHE'
            elif newdir == 'DROITE' and direction != 'GAUCHE':
                direction = 'DROITE'
            elif newdir == 'HAUT' and direction != 'BAS':
                direction = 'HAUT'
            elif newdir == 'BAS' and direction != 'HAUT':
                direction = 'BAS'
            
            deplacer_serpent(serpent, direction)
            fruit = manger_fruit(serpent, fruit)

            ecran.fill(noir)
            dessiner_serpent(serpent)
            dessiner_fruit(fruit)
            global nbrindividu
            afficher_texte("Génération: {} ; N°: {}".format(nbrgen_compteur, nbrindividu), 6, 6, blanc)
            # afficher_texte("Best Génération 142", 6, 6, blanc)
            # afficher_texte("Q-learning", 6, 6, blanc)
            pygame.display.update()
            temps_reinit +=1
            temps2 +=1

            temps_boucle = 40 
            if temps2 % temps_boucle == 0:
                repet0 = serpent[0]
                compteur_repetition=0
                tmp_mem = temps2
            if temps2 % temps_boucle == 1:
                repet1 = serpent[0]
            if temps2 % temps_boucle == 2:
                repet2 = serpent[0]   
            if temps2 % temps_boucle == 3:
                repet3 = serpent[0]       

            if serpent[0]== repet0 and temps2 > tmp_mem:
                is_repet1 = 1
                tmp_mem=temps2
            if is_repet1 == 1 and serpent[0]== repet1 and temps2== tmp_mem+1:
                is_repet2 = 1
                tmp_mem=temps2
            if is_repet2 == 1 and serpent[0]== repet2 and temps2== tmp_mem+1:
                is_repet3 = 1
                tmp_mem=temps2
            if is_repet3 == 1 and serpent[0]== repet3 and temps2== tmp_mem+1:
                compteur_repetition+=1
            
            death_mur=0
            death_corps=0
            boucle=0
            penalite=0
            if pommes != len(serpent)-1:
                avg_steps += temps_reinit
                temps_reinit =0 
                cmpt += 1
            pommes = len(serpent)-1
            
            
            if newdir != first_dir:
                no_change_dir = 0
            # else:
            #     count_same_direction = 0
             
            # else:
            #     scoregit += 2
            # if pommes==1 and temps2<= (hauteur/taille_case)*3:fg
            #         pommes=0    
            
            if collision_mur(serpent) or collision_serpent(serpent) or (temps_reinit==329): #  or (temps_reinit==300) 
                if collision_mur(serpent):
                    death_mur==1
                if collision_serpent(serpent):
                    death_corps==1
                if collision_mur(serpent) or collision_serpent(serpent):
                    death=1
                if temps_reinit==329:
                    boucle=1
                

                # if compteur_repetition==4:
                #     temps2+=240-temps_reinit

                if cmpt ==0:
                    avg_steps=0
                else:
                    avg_steps/=cmpt

                # if no_change_dir:
                #     pommes=0
                   
                # pygame.quit()
                # quit()
                
                # if temps2<=(hauteur/taille_case)-1:
                #     pommes=0    

                collision=1
                time.sleep(1 / vitesse)
                
                if modescore == 1:
                    return 5000*pommes + temps2  # + (2**pommes + (pommes**2.1)*5000) - ((pommes**1.2)*((0.25*temps2)**1.3)) 
            # t2=time.time()
                
            time.sleep(1/vitesse)
            # print(f"total boucle {t2-t1} ; precitmaxinn {t4-t3}")



def deplacer_serpent(serpent, direction):
    tete = list(serpent[0])

    if direction == 'DROITE':
        tete[0] += taille_case
    elif direction == 'GAUCHE':
        tete[0] -= taille_case
    elif direction == 'HAUT':
        tete[1] -= taille_case
    elif direction == 'BAS':
        tete[1] += taille_case

    serpent.insert(0, tete)
def generer_fruit2(serpent):
    x = round(random.randrange(0, largeur - taille_case) / taille_case) * taille_case
    y = round(random.randrange(0, hauteur - taille_case) / taille_case) * taille_case

    return [x, y]
def generer_fruit(serpent):
    available_positions = set()
    
    for x in range(0, largeur, taille_case):
        for y in range(0, hauteur, taille_case):
            position = [x, y]
            if position not in serpent:
                available_positions.add(tuple(position))

    return list(random.choice(list(available_positions)))
def dessiner_serpent(serpent):
    for i, partie in enumerate(serpent):
        if i == 0 or i == len(serpent)-1:
            pygame.draw.circle(ecran, vert, (partie[0] + (taille_case) // 2, partie[1] + (taille_case) // 2),( taille_case) // 2)

        else:
            pygame.draw.rect(ecran, vert, [partie[0], partie[1], taille_case, taille_case])
def dessiner_fruit(fruit):
    pygame.draw.rect(ecran, rouge, [fruit[0], fruit[1], taille_case, taille_case])
def manger_fruit(serpent, fruit):
    if serpent[0] == fruit:
        fruit = generer_fruit(serpent)
    else:
        serpent.pop()
    return fruit
def collision_mur(serpent):
    tete = serpent[0]
    return tete[0] < 0 or tete[0] >= largeur or tete[1] < 0 or tete[1] >= hauteur
def collision_serpent(serpent):
    tete = serpent[0]
    return tete in serpent[1:]
def collision_serpent2(serpent, voisin):
    tete = voisin
    return tete in serpent[:]
def aleat():
    return {0: 'GAUCHE', 1:'DROITE', 2:'HAUT', 3:'BAS'}.get(random.randint(0, 3))
def inverser(cle):
    if isinstance(cle, int):
        return {0: 'GAUCHE', 1:'DROITE', 2:'HAUT', 3:'BAS'}.get(cle)
    return {'GAUCHE':0, 'DROITE':1, 'HAUT':2, 'BAS':3}.get(cle)
def inverser2(cle):
    result = {0: 'GAUCHE', 1: 'DROITE', 2: 'HAUT', 3: 'BAS'}.get(cle)
    if result is not None:
        return result
    return {'GAUCHE': 0, 'DROITE': 1, 'HAUT': 2, 'BAS': 3}.get(cle)
def invmanuel(cle):
    return {pygame.K_UP:'HAUT', pygame.K_DOWN:'BAS', pygame.K_LEFT:'GAUCHE', pygame.K_RIGHT:'DROITE'}.get(cle)
def nombre_cases_accessibles(serpent):

    def dfs(x, y):

        visited.add((x, y))

        voisins = [(x + taille_case, y), (x - taille_case, y), (x, y + taille_case), (x, y - taille_case)]
        count = 1
        for voisin in voisins:
            if voisin not in visited and not collision_mur([voisin]) and not collision_serpent2(serpent,[voisin[0],voisin[1]]):
                count += dfs(voisin[0], voisin[1])
        
        return count
    
    visited = set()
    tete = serpent[0]
    accessible_count = dfs(tete[0], tete[1])
    
    return accessible_count
def dist(serpent, fruit):
    return abs(serpent[0][0]-fruit[0]) +abs(serpent[0][1]-fruit[1])
def recompense(serpent, serpent_copy, fruit):
    q_value = -100000*collision_mur(serpent_copy)-100000*collision_serpent(serpent_copy) + len(serpent_copy) - len(serpent) 
    q_value += 0.0000001 * ((dist(serpent, fruit)-dist(serpent_copy, fruit))>0) # pour l'approcher s'il est loin
    return q_value
def predictmaxi(serpent,fruit,olddirection, n):
    
    best_direction = None
    best_score = float('-inf')
    q_value=-100000
    best_score=-100000
    nbrcases = nombre_cases_accessibles(serpent)
    for action in range(4):  
        if possible(olddirection,inverser(action)):
            
            serpent_copy = serpent[:]
            fruit_copy = fruit[:]
            
            deplacer_serpent(serpent_copy, inverser(action))
            fruit_copy = manger_fruit(serpent_copy, fruit_copy)
            q_value = -100000*collision_mur(serpent_copy)-100000*collision_serpent(serpent_copy) + 1*(len(serpent_copy) - len(serpent)) 
            q_value += 0.0001 * ((dist(serpent, fruit)-dist(serpent_copy, fruit))>0) # pour l'approcher s'il est loin
            if n==1:
                
                #REMETTRE EN DESSOUS
                q_value += -10*(nbrcases > nombre_cases_accessibles(serpent_copy))
                # pass
            # if n<=1 and best_score <-1 and q_value>=0 :
            #     n=2
            if n>1 : # and q_value >=-1000  pour accelerer
                q_value += 0.93* predictmaxi(serpent_copy,fruit_copy,inverser(action), n-1)[1]
            
            
            if q_value > best_score:
                best_score = q_value
                best_direction = inverser(action)

    return best_direction,best_score
def possible(direction, newdir):
    ok=True
    if newdir == 'GAUCHE' and direction == 'DROITE':
            ok= False
    elif newdir == 'DROITE' and direction == 'GAUCHE':
        ok= False
    elif newdir == 'HAUT' and direction == 'BAS':
        ok= False
    elif newdir == 'BAS' and direction == 'HAUT':
        ok= False
    return ok
def distance_queue(serpent):
    tete = serpent[0]
    distance_obstacle=[hauteur]*8
    distance_obstacle[0] = tete[0]
    distance_obstacle[1] = largeur - tete[0] - taille_case
    distance_obstacle[2] = tete[1]
    distance_obstacle[3] = hauteur - tete[1] - taille_case
    distance_obstacle[4] = min((tete[0]), (tete[1])) * 2
    distance_obstacle[5] = min((tete[0]), (hauteur - tete[1]- taille_case)) * 2
    distance_obstacle[6] = min((largeur - tete[0] - taille_case), (tete[1])) * 2
    distance_obstacle[7] = min((largeur - tete[0]- taille_case), (hauteur - tete[1]- taille_case)) * 2
    # murG = tete[0]
    # murD = largeur - tete[0] - taille_case
    # murH = tete[1]
    # murB = hauteur - tete[1] - taille_case
    # distance_obstacle=[murG, murD, murH, murB, min(murH, murG)*2, min(murB, murG)*2, min(murH, murD)*2, min(murB, murD)*2]
    for partie in serpent[1:]:
        if tete[1] == partie[1] and tete[0] > partie[0]:  # GAUCHE
            distance_obstacle[0] = min(distance_obstacle[0], tete[0] - partie[0]- taille_case)
        if tete[1] == partie[1] and tete[0] < partie[0]:  # DROITE
            distance_obstacle[1] = min(distance_obstacle[1], partie[0] - tete[0]- taille_case)
        if tete[0] == partie[0] and tete[1] > partie[1]:  # HAUT
            distance_obstacle[2] = min(distance_obstacle[2], tete[1] - partie[1]- taille_case)
        if tete[0] == partie[0] and tete[1] < partie[1]:  # BAS
            distance_obstacle[3] = min(distance_obstacle[3], partie[1] - tete[1]- taille_case)
        if tete[0] > partie[0] and tete[1] > partie[1] and (tete[0] - partie[0] == tete[1] - partie[1]):  # HAUT_GAUCHE
            distance_obstacle[4] = min(distance_obstacle[4], min((tete[0] - partie[0]- taille_case), (tete[1] - partie[1]- taille_case)) * 2)
        if tete[0] > partie[0] and tete[1] < partie[1] and (tete[0] - partie[0] == - tete[1] + partie[1]):  # BAS_GAUCHE
            distance_obstacle[5] = min(distance_obstacle[5], min((tete[0] - partie[0]- taille_case), (partie[1] - tete[1]- taille_case)) * 2)
        if tete[0] < partie[0] and tete[1] > partie[1] and (- tete[0] + partie[0] == tete[1] - partie[1]):  # HAUT_DROITE
            distance_obstacle[6] = min(distance_obstacle[6], min((partie[0] - tete[0]- taille_case), (tete[1] - partie[1]- taille_case)) * 2)
        if tete[0] < partie[0] and tete[1] < partie[1] and (- tete[0] + partie[0] == - tete[1] + partie[1]):  # BAS_DROITE
            distance_obstacle[7] = min(distance_obstacle[7], min((partie[0] - tete[0]- taille_case), (partie[1] - tete[1]- taille_case)) * 2)
    distance_obstacle = [(elem/(hauteur-taille_case) if i <= 3 else elem/((hauteur-taille_case)*2)) for i, elem in enumerate(distance_obstacle)] 
    return distance_obstacle
def distance_mur(serpent):
    distance_mur=[0]*4
    tete = serpent[0]

    murG = tete[0] / (hauteur-taille_case)
    murD = (largeur - tete[0] - taille_case) / (hauteur-taille_case)
    murH = tete[1] / (hauteur-taille_case)
    murB = (hauteur - tete[1] - taille_case) / (hauteur-taille_case)
    distance_mur= [murG, murD, murH, murB] 
    return distance_mur
def distance_fruit(serpent, fruit):
    tete = serpent[0]
    dist=[hauteur]*8
    dist[0] = tete[0] - fruit[0]
    dist[1] = fruit[0] - tete[0]
    dist[2] = tete[1] - fruit[1]
    dist[3] = fruit[1] - tete[1]
    dist[4] = ((tete[0] - fruit[0])**2 + (tete[1] - fruit[1])**2)**0.5
    dist[5] = ((tete[0] - fruit[0])**2 + (fruit[1] - tete[1])**2)**0.5
    dist[6] = ((fruit[0] - tete[0])**2 + (tete[1] - fruit[1])**2)**0.5
    dist[7] = ((fruit[0] - tete[0])**2 + (fruit[1] - tete[1])**2)**0.5
    return dist
def if_fruit(serpent, fruit, direction):
    tete = serpent[0]
    dist=hauteur

    if direction == 'GAUCHE' and tete[1] == fruit[1] and tete[0] > fruit[0]:
        return 1
    elif direction == 'DROITE' and tete[1] == fruit[1] and tete[0] < fruit[0]:
        return 1
    elif direction == 'HAUT' and tete[0] == fruit[0] and tete[1] > fruit[1]:
        return 1
    elif direction == 'BAS' and tete[0] == fruit[0] and tete[1] < fruit[1]:
        return 1
    elif direction == 'HAUT_GAUCHE' and tete[0] > fruit[0] and tete[1] > fruit[1]:
        return 1
    elif direction == 'HAUT_DROITE' and tete[0] > fruit[0] and tete[1] < fruit[1]:
        return 1
    elif direction == 'BAS_GAUCHE' and tete[0] < fruit[0] and tete[1] > fruit[1]:
        return 1
    elif direction == 'BAS_DROITE' and tete[0] < fruit[0] and tete[1] < fruit[1]:
        return 1
    return 0
def if_queue(serpent, direction):
    tete = serpent[0]
    queue = serpent[1:]

    if direction == 'GAUCHE':
        for segment in queue:
            if segment[1] == tete[1] and segment[0] < tete[0]:
                return 1
    elif direction == 'DROITE':
        for segment in queue:
            if segment[1] == tete[1] and segment[0] > tete[0]:
                return 1
    elif direction == 'HAUT':
        for segment in queue:
            if segment[0] == tete[0] and segment[1] < tete[1]:
                return 1
    elif direction == 'BAS':
        for segment in queue:
            if segment[0] == tete[0] and segment[1] > tete[1]:
                return 1
    elif direction == 'HAUT_GAUCHE':
        for segment in queue:
            if segment[0] < tete[0] and segment[1] < tete[1]:
                return 1
    elif direction == 'HAUT_DROITE':
        for segment in queue:
            if segment[0] > tete[0] and segment[1] < tete[1]:
                return 1
    elif direction == 'BAS_GAUCHE':
        for segment in queue:
            if segment[0] < tete[0] and segment[1] > tete[1]:
                return 1
    elif direction == 'BAS_DROITE':
        for segment in queue:
            if segment[0] > tete[0] and segment[1] > tete[1]:
                return 1

    return 0
def direction_vers_fruit(serpent, fruit):
    tete = serpent[0]

    # Calcul des différences en coordonnées x et y entre la tête du serpent et le fruit
    diff_x = tete[0] - fruit[0]
    diff_y = tete[1] - fruit[1]

    # Choisir la direction en fonction des différences en coordonnées
    if abs(diff_x) > abs(diff_y):
        # La différence en coordonnée x est plus grande, donc on se déplace principalement en horizontal
        if diff_x > 0:
            return 'GAUCHE'
        else:
            return 'DROITE'
    else:
        # La différence en coordonnée y est plus grande, donc on se déplace principalement en vertical
        if diff_y > 0:
            return 'HAUT'
        else:
            return 'BAS'
def presence_obstacle(serpent):
    tete = serpent[0]
    obstacle_vector = [0, 0, 0, 0]  # Vecteur de dimension 4 initialement rempli de zéros

    if tete[0] == 0 or any(partie[0] == tete[0] - taille_case and partie[1] == tete[1] for partie in serpent):
        obstacle_vector[0] = 1  # GAUCHE
    if tete[0] == largeur - taille_case or any(partie[0] == tete[0] + taille_case and partie[1] == tete[1] for partie in serpent):
        obstacle_vector[1] = 1  # DROITE
    if tete[1] == 0 or any(partie[1] == tete[1] - taille_case and partie[0] == tete[0] for partie in serpent):
        obstacle_vector[2] = 1  # HAUT
    if tete[1] == hauteur - taille_case or any(partie[1] == tete[1] + taille_case and partie[0] == tete[0] for partie in serpent):
        obstacle_vector[3] = 1  # BAS

    return obstacle_vector

def presence_mur(serpent):
    tete = serpent[0]
    obstacle_vector = [0, 0, 0, 0]  # Vecteur de dimension 4 initialement rempli de zéros
    if tete[0] == 0:
        obstacle_vector[0] = 1  # GAUCHE
    if tete[0] == largeur - taille_case:
        obstacle_vector[1] = 1  # DROITE
    if tete[1] == 0:
        obstacle_vector[2] = 1  # HAUT
    if tete[1] == hauteur - taille_case:
        obstacle_vector[3] = 1  # BAS
    return obstacle_vector
def presence_queue(serpent):
    tete = serpent[0]
    obstacle_vector = [0, 0, 0, 0]  # Vecteur de dimension 4 initialement rempli de zéros
    if any(partie[0] == tete[0] - taille_case and partie[1] == tete[1] for partie in serpent):
        obstacle_vector[0] = 1  # GAUCHE
    if any(partie[0] == tete[0] + taille_case and partie[1] == tete[1] for partie in serpent):
        obstacle_vector[1] = 1  # DROITE
    if any(partie[1] == tete[1] - taille_case and partie[0] == tete[0] for partie in serpent):
        obstacle_vector[2] = 1  # HAUT
    if any(partie[1] == tete[1] + taille_case and partie[0] == tete[0] for partie in serpent):
        obstacle_vector[3] = 1  # BAS
    return obstacle_vector

def presence_queue_loin(serpent):
    tete = serpent[0]
    obstacle_vector = [0]*8  # Vecteur de dimension 4 initialement rempli de zéros
    if any(partie[0] <= tete[0] - taille_case and partie[1] == tete[1] for partie in serpent):
        obstacle_vector[0] = 1  # GAUCHE
    if any(partie[0] >= tete[0] + taille_case and partie[1] == tete[1] for partie in serpent):
        obstacle_vector[1] = 1  # DROITE
    if any(partie[1] <= tete[1] - taille_case and partie[0] == tete[0] for partie in serpent):
        obstacle_vector[2] = 1  # HAUT
    if any(partie[1] >= tete[1] + taille_case and partie[0] == tete[0] for partie in serpent):
        obstacle_vector[3] = 1  # BAS

    if any(partie[0] <= tete[0] - taille_case and partie[1] <= tete[1] - taille_case and tete[0] - taille_case - partie[0] == tete[1] - taille_case - partie[1] for partie in serpent):
        obstacle_vector[4] = 1  # HAUT_GAUCHE
    if any(partie[0] >= tete[0] + taille_case and partie[1] <= tete[1] - taille_case and - tete[0] - taille_case + partie[0] == tete[1] - taille_case - partie[1] for partie in serpent):
        obstacle_vector[5] = 1  # HAUT_DROITE
    if any(partie[0] <= tete[0] - taille_case and partie[1] >= tete[1] + taille_case and tete[0] - taille_case - partie[0] == - tete[1] - taille_case + partie[1] for partie in serpent):
        obstacle_vector[6] = 1  # BAS_GAUCHE
    if any(partie[0] >= tete[0] + taille_case and partie[1] >= tete[1] + taille_case and - tete[0] - taille_case + partie[0] == - tete[1] - taille_case + partie[1] for partie in serpent):
        obstacle_vector[7] = 1  # BAS_DROITE
    return obstacle_vector


def direction_fruit_vecteur(serpent, fruit):
    tete = serpent[0]

    diff_x = fruit[0] - tete[0]
    diff_y = fruit[1] - tete[1]

    direction_vector = [0, 0, 0, 0]

    if diff_x > 0:
        direction_vector[0] = 1  # Fruit à droite
    elif diff_x < 0:
        direction_vector[1] = 1  # Fruit à gauche

    if diff_y > 0:
        direction_vector[2] = 1  # Fruit en bas
    elif diff_y < 0:
        direction_vector[3] = 1  # Fruit en haut

    return direction_vector
def sbx_crossover(parent1, parent2, eta=1.0):
    # SBX crossover for continuous variables
    child1 = np.zeros_like(parent1)
    child2 = np.zeros_like(parent2)

    for i in range(len(parent1)):
        if np.random.rand() <= 0.5:
            beta = (2.0 * np.random.rand()) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - np.random.rand()))) ** (1.0 / (eta + 1.0))

        child1[i] = 0.5 * (((1 + beta) * parent1[i]) + (1 - beta) * parent2[i])
        child2[i] = 0.5 * (((1 - beta) * parent1[i]) + (1 + beta) * parent2[i])

    return child1, child2
def crossover2(parent1, parent2):

    crossover_point = random.randint(0, len(parent1)-1)
    
    for i in range(0, len(parent1), 2):
        
        crossover_point = random.randint(1, len(parent1[i])-1)

        parent1[i] = np.array(parent1[i][:crossover_point].tolist() + parent2[i][crossover_point:].tolist())
        parent2[i] = np.array(parent2[i][:crossover_point].tolist() + parent1[i][crossover_point:].tolist())
        crossover_point2 = random.randint(1, len(parent1[i][crossover_point])-1)
        parent1[i][crossover_point-1] = np.array(parent1[i][crossover_point-1][crossover_point2:].tolist() + parent2[i][crossover_point-1][:crossover_point2].tolist())
        parent2[i][crossover_point-1] = np.array(parent1[i][crossover_point-1][:crossover_point2].tolist() + parent2[i][crossover_point-1][crossover_point2:].tolist())

    return parent1, parent2

def crossover_fast(parent1, parent2):
    w1= parent1.get_weights()
    w2= parent2.get_weights()
    for i in range(0, len(w1), 2):
        for j in range(0, len(w1[i])):
            for k in range(0, len(w1[i][j])):
                nbr=random.uniform(0, 1)
                if nbr < 0.5:
                    w1[i][j][k]=w2[i][j][k]
    parent1.set_weights(w1)
    return parent1

def crossover(parent1, parent2, proba_echange):
    enfant1 = copy.deepcopy(parent1)
    enfant2 = copy.deepcopy(parent2)
    for i in range(0, len(parent1), 2):
        for j in range(0, len(parent1[i])):
            for k in range(0, len(parent1[i][j])):
                nbr=random.uniform(0, 1)
                if nbr < proba_echange:
                    enfant1[i][j][k]=parent2[i][j][k]
                    enfant2[i][j][k]=parent1[i][j][k]

    for i in range(1, len(parent1)+1, 2): #couches des biais
        for j in range(0, len(parent1[i])):
            nbr=random.uniform(0, 1)
            if nbr < proba_echange:
                enfant1[i][j]=parent2[i][j]
                enfant2[i][j]=parent1[i][j]

    return enfant1, enfant2


def crossover_1_point(parent1, parent2):
    save = copy.deepcopy(parent1)
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    crossover_point = random.randrange(0, len(parent1) - 1, 2)
    crossover_point2 = random.randint(0, len(parent1[crossover_point]) - 1)
    crossover_point3 = random.randint(0, len(parent1[crossover_point][crossover_point2]) - 1)
    
    for i in range(crossover_point, len(parent1), 2):
        for j in range(crossover_point2, len(parent1[i])):
            for k in range(crossover_point3, len(parent1[i][j])):
                save[i][j][k] = parent1[i][j][k]
                child1[i][j][k]=child2[i][j][k]
                child2[i][j][k]=save[i][j][k]
    return child1,child2

def crossover_ameliore(parent1, parent2):
    crossover_point = random.randint(0, len(parent1)-1)
    
    parent1_array = np.array(parent1)
    parent2_array = np.array(parent2)

    mask = np.random.rand(*parent1_array.shape) < 0.5
    offspring = np.where(mask, parent1_array, parent2_array)

    return offspring.tolist()
def evaluer_individu(individu, passages=1):
    global nbrindividu 
    model.set_weights(individu.get_weights())
    score_min = 10000000000000000
    for i in range(passages):
        score = jeu(modescore=1, bool_random=0, mode="nn")  # Remplacez par la logique d'évaluation du jeu
        if score_min > score:
            score_min = score
    nbrindividu += 1
    return score_min
class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_percentage, passages, proba_echange, force_mutation):
        self.population_size = population_size
        self.proba_echange = proba_echange
        self.mutation_rate = mutation_rate
        self.force_mutation = force_mutation
        self.crossover_percentage = crossover_percentage
        self.scoremoyen = 0
        self.bestscore= 0
        self.passages= passages
        # self.reseaux_temp =  [create_nn() for _ in range(population_size)]
        self.reseau = create_nn()
        self.poids = []
    def evolve(self, current_population):
        # for i in range(self.population_size):
        #     self.reseaux_temp[i].set_weights(current_population[i])
        # scores = np.array([evaluer_individu(individu, passages=1) for individu in self.reseaux_temp])
 
        scores=[]
        for i in range(self.population_size):
            self.reseau.set_weights(current_population[i])
            scores.append(evaluer_individu(self.reseau, passages=self.passages))

        nombre_best_choose = 1
        nombre_passage = 4 # multiplié par le nbr de passages de base
        self.scoremoyen = sum(scores) / self.population_size
        indices_bests = [i for i in np.argsort(scores)[::-1][:nombre_best_choose]]
        
        for p in range(len(indices_bests)):
            scores_pretendant = []
            self.reseau.set_weights(current_population[indices_bests[p]])
            for j in range(nombre_passage):
                scores_pretendant.append(evaluer_individu(self.reseau, passages=self.passages))
                
            if min(scores_pretendant) > self.bestscore:
                filename = f"best.h5"
                self.reseau.save(filename)
                self.bestscore = min(scores_pretendant)
        
        # indice_best = argmax(scores)
        if nbrgen_compteur < 1000000000 :
            selected_indices = self.selection(scores, self.crossover_percentage)
            new_poids = self.reproduction_test(current_population, selected_indices)
        else:
            selected_indices = self.selection1(scores, self.crossover_percentage)
            new_poids = self.reproduction1(current_population, selected_indices)
        new_poids = self.mutation(new_poids, self.mutation_rate)

        return new_poids

    def selection1(self, scores, top_percentage):
        total_fitness = np.sum(scores)
        probabilities = scores / total_fitness
        indices= []
        selected_indices = np.random.choice(range(len(scores)), size=self.population_size, p=probabilities)
        for i in range(len(scores)):
            scores_sans = copy.deepcopy(selected_indices)
            scores_sans[i] = 0
            total_fitness2 = np.sum(scores_sans)
            probabilities2 = scores_sans / total_fitness2
            selected_indices2 = np.random.choice(range(len(scores_sans)), size=1, p=probabilities2)
            indices.append([selected_indices[i], selected_indices2[0]])
        return indices
    
    def selection2(self, scores, top_percentage):
        ranked_indices = np.argsort(scores)
        selection_probabilities = np.arange(1, len(scores) + 1) / np.sum(np.arange(1, len(scores) + 1))
        selected_indices = np.random.choice(ranked_indices, size=self.population_size, p=selection_probabilities)
        return selected_indices
    def selection(self, scores, top_percentage):
        sorted_indices = np.argsort(scores)
        num_top_candidates = int(top_percentage * len(scores))
        selected_indices = sorted_indices[-num_top_candidates:]
        return selected_indices

    def reproduction(self, current_population, selected_indices):
        parents = [current_population[i] for i in selected_indices]
        new_poids = []
        # weights = np.arange(1, len(selected_indices) + 1)
        for i in range(self.population_size // 2):
            # ind1,ind2 = np.random.choice(list(range(len(selected_indices))) , size=2, replace=False, p=weights/weights.sum())
            ind1,ind2 = np.random.choice(list(range(len(selected_indices))) , size=2, replace=False)
            parent1, parent2 = parents[ind1], parents[ind2]
            child, child2 = crossover(parent1, parent2, self.proba_echange)
            new_poids.append(child)
            new_poids.append(child2)

        return new_poids
    
    def reproduction_test(self, current_population, selected_indices):
        parents = [current_population[i] for i in selected_indices]
        new_poids = []
        selected_couples = set()
        for i in range(self.population_size // 2):
            j=0
            while True:
                ind1, ind2 = np.random.choice(list(range(len(selected_indices))), size=2, replace=False)

                if (ind1, ind2) not in selected_couples and (ind2, ind1) not in selected_couples:
                    selected_couples.add((ind1, ind2))
                    break
                j+=1
                if j==2000:
                    break
            parent1, parent2 = parents[ind1], parents[ind2]
            child, child2 = crossover(parent1, parent2, self.proba_echange)
            new_poids.append(child)
            new_poids.append(child2)

        return new_poids
    
    def reproduction1(self, current_population, selected_indices):
        new_poids = []
        # weights = np.arange(1, len(selected_indices) + 1)
        for indices in selected_indices:
            parent1, parent2 = current_population[indices[0]], current_population[indices[1]]
            child, child2 = crossover(parent1, parent2, self.proba_echange)
            new_poids.append(child)
            new_poids.append(child2)

        return new_poids

    def reproduction5(self, current_population, selected_indices):
        
        parents = [current_population[i] for i in selected_indices]
        new_population = []
        for _ in range(self.population_size):
            
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)

            child_weights = []  
            for layer_parent1, layer_parent2 in zip(parent1.get_weights(), parent2.get_weights()):
                # crossover_point = np.random.randint(2, size=np.shape(layer_parent1)).astype(bool)
                # child_weights.append(np.where(crossover_point, layer_parent1, layer_parent2))
                
                crossover_mask = np.random.choice([0, 1], size=np.shape(layer_parent1), p=[0.5, 0.5])
                child_weights.append(np.where(crossover_mask, layer_parent1, layer_parent2))

            child = create_nn()
            child.set_weights(child_weights)

            new_population.append(child)

        return new_population
    
    def mutation1(self, new_population, mutation_rate=0.01):
        for child_weights in new_population:
            if random.random() < mutation_rate:
                for i in range(len(child_weights)):
                    # mutation = np.random.normal(loc=0, scale=0.1, size=child_weights[i].shape)
                    
                    jj = random.choice(range(0, len(child_weights)))
                    j = random.choice(range(0, len(child_weights[jj])))
                    
                    jj2 = random.choice(range(0, len(child_weights)))
                    j2 = random.choice(range(0, len(child_weights[jj])))

                    # jj2 = random.choice(range(0, len(child_weights), 2))
                    # j2 = random.choice(range(0, len(child_weights[jj2])))
                    # k2 = random.choice(range(0, len(child_weights[jj2][j2])))

                    # mutation = np.random.choice(np.arange(-1, 1, step = 0.001), size = (1), replace = False)     
                    # child_weights[jj][j][k] += np.random.normal(loc=0, scale=abs(child_weights[jj2][j2][k2])/10)
                    # save = child_weights[jj][j][k]      
                    # mutation = random.randint(-50, 50)/1000
                    if jj%2 == 0:
                        k = random.choice(range(0, len(child_weights[jj][j])))
                        k2 = random.choice(range(0, len(child_weights[jj2][j2])))
                        child_weights[jj][j][k] += np.random.normal(loc=0, scale=abs(child_weights[jj2][j2][k2]))   # mutation * child_weights[jj][j][k]
                    else:
                        child_weights[jj][j] += np.random.normal(loc=0, scale=abs(child_weights[jj2][j2]))  # mutation * child_weights[jj][j]
                    # child_weights[jj2][j2][k2] =  np.random.choice([-1, 1])*np.random.uniform(0.5, 2)*save
            
        return new_population

    def mutation(self, new_population, mutation_rate=0.01):
        for child_weights in new_population:
                for i in range(len(child_weights)):
                    for ii in range(len(child_weights[i])):
                        if random.random() < mutation_rate and i%2 == 1:
                            mutation = random.randint(- self.force_mutation, self.force_mutation)/1000     
                            child_weights[i][ii] += mutation * child_weights[i][ii]
                        if i%2 == 0:
                            for j in range(len(child_weights[i][ii])):
                                if random.random() < mutation_rate:
                                    mutation = random.randint(- self.force_mutation, self.force_mutation)/1000                                
                                    child_weights[i][ii][j] += mutation * child_weights[i][ii][j]
                        
            
        return new_population
    
def train_genetic():
    population_size = 80
    nbr_generation=200
    multiplicateur = 1

    crossover_percentage = 0.10
    proba_echange = 0.1

    mutation_rate = 0.08
    force_mutation = 500  # base 1000

    passages=2

    
    genetic_algorithm = GeneticAlgorithm(population_size*multiplicateur, mutation_rate, crossover_percentage, passages, proba_echange, force_mutation)
    t1=time.time()
    initial_population  = [create_nn() for _ in range(population_size)]

# # a enlever apres
    for i in range(population_size):
        initial_population[i].load_weights(f"individual_{i+1}.h5")
    # initial_population[0].load_weights(f"best210.h5")
    # initial_population[1].load_weights(f"best409.h5")
    # initial_population[2].load_weights(f"best460.h5")
    # initial_population[3].load_weights(f"best460.h5")
    


    poids = []
    for i in range(population_size):
        poids.append(initial_population[i].get_weights())
 
    for i in range(population_size*(multiplicateur-1)):
        new_poids = copy.deepcopy(poids[0])        
        for i in range(0, len(poids[0]), 2):
            for j in range(0, len(poids[0][i])):
                for k in range(0, len(poids[0][i][j])):
                    new_poids[i][j][k] = np.random.choice([np.random.choice([-1, 1]), 1], p=[0.2, 0.8]) * np.random.uniform(0.769, 1.3)*poids[random.randint(0,population_size-1)][i][j][k] # 
        poids.append(new_poids)


    # for i in range(population_size):
    #     nouveau_vecteur = [np.random.rand(*arr.shape).astype(np.float32) for arr in initial_population[0].get_weights()]
    #     poids.append(nouveau_vecteur)

    maxscore=0
    for generation in range(nbr_generation):
        global nbrgen_compteur 
        global nbrindividu
        nbrindividu = 1
        nbrgen_compteur += 1
        
        # next_population = [create_nn() for _ in range(population_size)]
        
        poids = genetic_algorithm.evolve(poids)
        print(f"Generation {generation + 1} : {genetic_algorithm.scoremoyen} /// Best : {genetic_algorithm.bestscore}")
        if generation in [nbr_generation-1]:
            for i, individual in enumerate(poids):
                model.set_weights(individual)
                filename = f"individual_{i + 1}.h5"
                model.save(filename)
        # K.clear_session()

        if generation in [-1]:
            
            for i, individual in enumerate(poids):  
                model.set_weights(individual)
                score=evaluer_individu(model)
                if score>maxscore:
                    maxscore=score
                    poids_to_save = copy.deepcopy(model.get_weights())
            filename = f"best.h5"
            model.set_weights(poids_to_save)
            model.save(filename)
    
train_genetic()

# jeu(modescore=0, bool_random=0, mode="maxi")

# model.load_weights("best460.h5")
# evaluer_individu(model, passages=100)




