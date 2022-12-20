import numpy as np
import cv2

import math
import random
from random import choice

from gym import Env, spaces

#### COLORS (BGR) ####
BLUE  = [255,   0,   0]
GREEN = [0,   255,   0]
RED   = [0,     0, 255]
WHITE = [255, 255, 255]
GREY  = [240, 240, 240]
BLACK = [0,     0,   0]
CYAN  = [255, 255,   0]
#### BUILDINGS #######
WASTELAND = 0
OFFICE    = 1
HOUSE     = 2
PARK      = 3
COM       = 4
######################

class City(Env):
    def __init__(self, mapshape = (10, 10), start_shape = (3, 3), path='./'):
        super(City, self).__init__()
        self.path=path
        #### reset le programme quand il reste STOP cases vides
        self.stop = max(mapshape)*4
        # on initialise la map memorisant l'environement
        self.mapshape = mapshape
        # on definit la taille de l'espace d'observation et l'espace en question
        self.observation_shape = (7,7)
        self.observation_space = spaces.Box(low=0, high=5, shape=self.observation_shape, dtype = int)
        #on definit l'espace de depart qui sera genere aleatoirement
        self.start_shape = start_shape
        self.start_size = start_shape[0]
        #on definit la taille de l'image qui representera l'environement
        self.canvas_shape = 700, 700, 3 # width, height, color (BGR)
        #on definit la variable qui representera notre environement
        self.canvas = np.ones(self.canvas_shape, dtype = np.uint8) * 0
        #on definit le nombre d'actions (ici 4)
        self.action_space = spaces.Discrete(4)
        #on initialise la somme de la recompense cumule sur les episodes.
        self.sum = 0
        pass
    
    # fonction qui permet de redemarrer l'environnement
    def reset(self, random_start = True):
        # on sauvegarde la somme des recompenses cumule dans une fichier
        with open(str(self.path)+str('rewardDQN.txt'), "a+") as fhandle:
            fhandle.write(str(self.sum)+'\n')
        #on reinitialise la somme des recompenses
        self.sum = 0
        # on redefinit la position du joueur au milieu de la carte
        self.position = (self.mapshape[0] // 2), (self.mapshape[1] // 2)
        # on reinitialise la carte du jeu
        self.map = np.ones(self.mapshape, dtype = np.uint8) * WASTELAND
        self.offices = []
        self.houses = []
        self.parks = []
        self.coms = []
        self.adjacents_cells = {}
        #on reinitialise certaines variables
        self.loop_number = 0
        self.start_size = self.start_shape[0]
        self.actual_size = 0
        self.actual_region = 0
        
        # on replace des batiments au mileu de la carte de maniere aleatoire
        start_shape = self.start_shape
        if random_start : # si le mode aleatoire est actif
            maisonX=([i for i in range((self.mapshape[1] - start_shape[1]) // 2, (self.mapshape[1] + start_shape[1]) // 2) if i !=(self.mapshape[1] // 2) ])
            maisonY=choice([i for i in range((self.mapshape[0] - start_shape[0]) // 2, (self.mapshape[0] + start_shape[0]) // 2) if i != (self.mapshape[0] // 2)])
            oficeX=choice([i for i in range((self.mapshape[1] - start_shape[1]) // 2, (self.mapshape[1] + start_shape[1]) // 2) if (i != maisonX) and (i!=(self.mapshape[1] // 2))])
            oficeY=choice([i for i in range((self.mapshape[0] - start_shape[0]) // 2, (self.mapshape[0] + start_shape[0]) // 2) if (i != maisonY) and (i!=(self.mapshape[0] // 2))])
            # on applique la generation pour chaque case au centre de la carte
            for y in range((self.mapshape[1] - start_shape[1]) // 2, (self.mapshape[1] + start_shape[1]) // 2):
                for x in range((self.mapshape[0] - start_shape[0]) // 2, (self.mapshape[0] + start_shape[0]) // 2):
                    # on selection le batiment a poser de maniere aleatoire
                    g=random.randrange(2) + 1
                    if (y!=(self.mapshape[0] // 2) or (x!=(self.mapshape[1] // 2))):
                        self.map[y, x] = g
                        if x==maisonX and y==maisonY:
                            self.map[y, x] = HOUSE
                            self.houses.append((y, x))
                        elif x==oficeX and y==oficeY:
                            self.map[y, x] = OFFICE
                            self.offices.append((y, x))
                        elif self.map[y, x] == OFFICE : self.offices.append((y, x))
                        elif self.map[y, x] == HOUSE  : self.houses.append((y, x))
                        self.delete_cell((y, x))
                        self.mark_adjacents_cells((y, x))
        x0 = self.position[0]
        y0 = self.position[1]
        OBSMAP = self.getMap(self.map,x0,y0)
        self.vus=OBSMAP
        return OBSMAP

    # fonction qui retourne l'espace d'observation associe a l'environement et la position de l'agent
    def getMap(self,mape,x,y):
        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 10)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value
        mape=np.pad(mape, 3, pad_with, padder=0)
        x=x+3
        y=y+3
        return mape[x-3:x+4,y-3:y+4]    
    
    # fonction qui retourne la distance entre une position et le bureau le plus proche de la position, avec une valeur maximale de 3
    # on estime qu'apres 3, la distance a beaucoup moins d'importance relative
    def __search_nearest_office(self, position):
        a=[math.dist(position, office) for office in self.offices]
        if 0 in a:
            a.remove(0)
        if min(a)>=2.9:
            return 1000
        else:
            return min(a)
        

    # fonction qui retour la distance entre une position et le parc le plus proche de la position, avec une valeur maximale de 3
    # (permet d'eviter les bugs de distance infini s'il n'y a pas de parc et aussi parce qu'il ne serait pas present dans l'espace d'observation)
    def __search_nearest_Park(self, position):
        a=[math.dist(position, office) for office in self.parks]
        if 0 in a:
            a.remove(0)
        a.append(3)
        if min(a)>=2.9:
            return 1000
        else:
            return min(a)
    
    # fonction qui retourne la distance entre une position et le bureau le plus proche de la position, avec une valeur maximale de 3
    def __search_nearest_house(self, position):
        a=[math.dist(position, house) for house in self.houses]
        if 0 in a:
            a.remove(0)
        if min(a)>=2.9:
            return 1000
        else:
            return min(a)

    # fonction qui retourne la distance moyenne d'un type de batiment avec le centre de la map en parametre, avec une valeur maximale de 3
    def __meanDistance(self ,maap, Type):
        coord=np.argwhere(np.array(maap) == Type)
        if len(coord)>0:
            return np.mean(np.array([math.dist((2,2), truc) for truc in coord]))
        else:
            return 1000
    
    # fonction qui renvoit -4 s'il y a plus de 30% de bureaux sur les cases non vides
    # sinon 4 dans le rayon de consideration (5 par 5)
    def __isTooMutchOffice(self, maap, Type):
        coord=len(np.argwhere(np.array(maap) == Type))
        coord0=len(np.argwhere(np.array(maap) == 0))
        if coord>(int((25-coord0)*0.3)+1):
            return 4
        else:
            return -4
    
    # fonction qui retourne le nombre de maisons adjacentes
    def __NbmaisonAdjasante(self, maap):
        maape=np.array(maap)
        maape=maape[1:4,1:4]
        coord=np.argwhere(np.array(maape) == HOUSE)
        return len(coord)
      
    # fonction qui retourne le nombre de maisons dans un buffer de 2
    def __NbmaisonAdjasantePARK(self, maap):
        maape=np.array(maap)
        coord=np.argwhere(np.array(maape) == HOUSE)
        return len(coord) 

    # fonction qui evalue si une case est vide (ne contient pas de batiment)
    def __is_free(self, position):
        return self.map[position] == WASTELAND
    
    # fonction qui supprime une cellule de la liste des cellules adjacentes
    def delete_cell(self, position):
        try : del self.adjacents_cells[position]
        except KeyError : pass
    
    # fonction qui incremente de 1 la valeur adjacente d'une cellule adjacente
    # la valeur adjacente correspond au nombre de batiment adjacent a la cellule en question (maximum de 8 du coup)
    def mark_cell(self, position):
        y, x = position
        # verifie que la cellule n'est pas hors carte et n'est pas deja dans une categorie de batiment
        if x < 0 or x >= self.mapshape[0] : return
        if y < 0 or y >= self.mapshape[1] : return
        if tuple(position) in self.houses : return
        if tuple(position) in self.offices : return
        if tuple(position) in self.parks : return
        if tuple(position) in self.coms : return
        try :
            self.adjacents_cells[tuple(position)] += 1
        except KeyError :
            self.adjacents_cells[tuple(position)] = 1
    
    # fonction qui incremente la valeur adjacente de toutes les cellules adjacentes
    def mark_adjacents_cells(self, position):
        y, x = position
        for position in [[y - 1, x - 1], [y - 1, x], [y - 1, x + 1], [y, x - 1], [y, x + 1], [y + 1, x - 1], [y + 1, x], [y + 1, x + 1]] :
            self.mark_cell(position)
        pass

    # fonction qui retourne la recompense associe a une position, une action et un batiment
    # si l'evaluation est negative, alors met a jour les memoires
    def q__place(self,position,inpute, is_placing_house,evaluation):
        # on prend une zone plus proche pour evaluer la recompense
        inpute=inpute[3-2:3+3,3-2:3+3]

        # si l'action est une maison
        if is_placing_house==1:
            r1 = self.__search_nearest_office(position)
            r1=1/(math.sqrt(r1**2))
            r2=self.__NbmaisonAdjasante(inpute)
            r2=r2/8
            r3=self.__meanDistance(inpute,COM)
            r3=1/(math.sqrt(r3**2))
            r5=self.__search_nearest_Park(position)
            r5=(1/r5)
            reward =(5*r1+r2+7*r3+7*r5)
            if not evaluation:
                self.houses.append(self.position)
                self.map[self.position] = HOUSE            
        
        # si l'action est un bureau
        elif is_placing_house==0 :  
            r1=self.__meanDistance(inpute,HOUSE)
            r1=1/(math.sqrt(r1**2))  
            r2 = self.__search_nearest_office(position)
            r2=1/(math.sqrt(r2**2))
            div=self.__isTooMutchOffice(inpute,OFFICE)
            reward =(5*r1+r2)-(div)
            if not evaluation:    
                self.offices.append(self.position)
                self.map[self.position] = OFFICE    

        # si l'action est un parc
        elif is_placing_house==2:
            r1=self.__NbmaisonAdjasantePARK(inpute)
            reward=r1/5
            if not evaluation:
                self.parks.append(self.position)
                self.map[self.position] = PARK
        
        # si l'action est un centre commercial
        elif is_placing_house==3:
            r1=self.__NbmaisonAdjasante(inpute)
            reward=r1/1    
            if not evaluation:
                self.coms.append(self.position)
                self.map[self.position] = COM

        # autre cas (impossible)
        else: reward = 0
            
        if not evaluation:
            self.delete_cell(self.position)
            self.mark_adjacents_cells(self.position)
            self.sum=self.sum+reward    
        
        return reward

    # fonction obsolete (actuellement)
    def GetValue(self):
        array=0
        def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 10)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value
        for x in range(max(self.mapshape)):
            for y in range(max(self.mapshape)):
                inpute=np.pad(self.map, 3, pad_with, padder=0)
                x1=x+3
                y1=y+3
                inpute=inpute[x1-3:x1+4,y1-3:y1+4]
                inpute[3,3]=0
                value=self.q__place((x,y),inpute, (self.map[x,y])-1,True)
                array=array+value
        # on sauvegarde        
        with open(str(self.path)+str('rewardTotalDQN.txt'), "a+") as fhandle:
            fhandle.write(str(array)+'\n')       
            
    
    # fonction qui evalue si une position est hors de la carte
    def __is_oob(self, position):
        return not(0 <= position[0] < self.mapshape[0]) \
            or not(0 <= position[1] < self.mapshape[1])
    
    # fonction qui selectionne la cellule du prochain batiment a poser
    # parmis les cellules ayant une valeur adjacent superieure ou egale a 2
    def select_random_cell(self):
        for position in self.adjacents_cells :
            if self.adjacents_cells[position] >= 2 :
                # retourne la premiere cellule trouve ayant 2 cellules adjacente contenant des batiments
                return position
        # si aucune cellule n'est trouve respectant les criteres precedent, retourne la plus ancienne cellule ayant un batiment adjacent 
        return self.adjacents_cells[0]
    
    # fonction qui selectionne la cellule du prochain batiment a poser
    # en suivant une forme de spirale
    def select_next_cell(self):
        # obtient la position de depart qui represente la cellule en haut a gauche de la spirale
        start_position = (self.mapshape[1] - self.start_size) // 2, (self.mapshape[0] - self.start_size) // 2
        start_position = start_position[0] - self.actual_size, start_position[1] - self.actual_size
        
        # continue la spirale en fonction de la region de la position actuelle
        # si la position actuelle est a l'ouest
        if self.actual_region == 0 :
            position = (start_position[0] - 1,
                        start_position[1])
            position = (position[0],
                        position[1] + self.loop_number)
        # si la position actuelle est au sud
        elif self.actual_region == 1 :
            position = (start_position[0],
                        start_position[1] + self.start_size + self.actual_size * 2)
            position = (position[0] + self.loop_number,
                        position[1])
        # si la position actuelle est a l'est
        elif self.actual_region == 2 :
            position = (start_position[0] + self.start_size + self.actual_size * 2,
                        start_position[1] - 1 + self.start_size  + self.actual_size * 2)
            position = (position[0],
                        position[1] - self.loop_number)
        # si la position actuelle est au nord
        elif self.actual_region == 3 :
            position = (start_position[0] - 1 + self.start_size + self.actual_size * 2,
                        start_position[1] - 1)
            position = (position[0] - self.loop_number,
                        position[1])
        
        # sauvegarde la position future
        self.loop_number += 1
        # si la position a depasse la taille de la region
        if self.loop_number > self.start_size + self.actual_size * 2 :
            # reinitialise la position pour la nouvelle region
            self.loop_number = 0
            
            # change de region
            self.actual_region += 1
            # si toutes les regions ont ete remplies 1 fois
            if self.actual_region >= 4 :
                # retourne a la premiere region (ouest)
                self.actual_region = 0
                # augmente la taille de la spirale
                self.actual_size += 1
            
        # retourne la position actuelle
        return position
    
    # fonction qui appele pour chaque action
    def step(self, action):
        reward = self.q__place(self.position,self.vus,action,False) # 1 = HOUSE / 0 = OFFICE
        self.reward = reward
        self.draw_elements_on_canvas()
        # on reinitialise s'il ny a plus assez d'espace vacant dans la ville
        if np.count_nonzero(self.map==0)<self.stop:
            self.position = self.select_next_cell()
            x0 = self.position[0]
            y0 = self.position[1]
            OBSMAP = self.getMap(self.map,x0,y0)
            self.vus=OBSMAP
            # on sauvegarde le nombre de chaque batiment predit pour avoir une idee de la diversiter predite
            conteurDeDiversiter=[]
            conteurDeDiversiter.append(len(self.offices))
            conteurDeDiversiter.append(len(self.houses))
            conteurDeDiversiter.append(len(self.parks))
            conteurDeDiversiter.append(len(self.coms))
            with open(str(self.path)+str('DivDqn.txt'), "a+") as fhandle:
                fhandle.write(str(conteurDeDiversiter)+'\n')
            # on sauvegarde la recompense totale sur toute la ville
            # self.GetValue()

            # retourne l'espace d'observation, la recompense, si l'environnement est termine et les informations supplementaire
            return OBSMAP, reward, True, {}

        self.position = self.select_next_cell()
        x0=self.position[0]
        y0=self.position[1]
        OBSMAP=self.getMap(self.map,x0,y0)
        self.vus=OBSMAP

        # retourne l'espace d'observation, la recompense, si l'environnement est termine et les informations supplementaire
        return OBSMAP, reward, False, {}
    
    # fonction qui dessine un batiment sur le canvas
    def __draw_element_on_canvas(self, y, x, color):
        observation_width, observation_height = self.mapshape
        canvas_width, canvas_height, _ = self.canvas_shape

        drawing_width = int(canvas_width / observation_width)
        drawing_height = int(canvas_height / observation_height)

        # adapte le batiment au canvas (en fonction de la taille du batiment et du canvas)
        for j in range(y * drawing_height, y * drawing_height + drawing_height):
            for i in range(x * drawing_width, x * drawing_width + drawing_width):
                try : self.canvas[i, j] = color
                except IndexError : 
                    print('error')
                    pass
                
        for j in range(y * drawing_height, y * drawing_height + drawing_height):
            try : self.canvas[x * drawing_width, j] = GREY
            except IndexError : pass
            
            
        for i in range(x * drawing_width, x * drawing_width + drawing_width):
            try : self.canvas[i, y * drawing_height] = GREY
            except IndexError : pass
        pass

    # fonction qui dessine le carre representant la zone d'observation
    def __draw_area_position(self, thickness = 3): # thickness doit etre un entier impair
        SIZE = self.observation_shape[0] // 2
        
        y, x = self.position
        thickness_range = range(- (thickness // 2), thickness // 2 + 1)
        
        observation_width, observation_height = self.mapshape
        canvas_width, canvas_height, _ = self.canvas_shape

        drawing_width = int(canvas_width / observation_width)
        drawing_height = int(canvas_height / observation_height)
        
        for j in range((y - SIZE) * drawing_height, (y + SIZE) * drawing_height + drawing_height):
            try :
                for t in thickness_range:
                    self.canvas[(x - SIZE) * drawing_width + t, j] = BLACK
                    self.canvas[(x + SIZE + 1) * drawing_width + t, j] = BLACK
            except IndexError : pass

        for i in range((x - SIZE) * drawing_width, (x + SIZE) * drawing_width + drawing_width):
            try :
                for t in thickness_range:
                    self.canvas[i, (y - SIZE) * drawing_height + t] = BLACK
                    self.canvas[i, (y + SIZE + 1) * drawing_height + t] = BLACK
            except IndexError : pass
            
        pass
    
    # fonction qui dessine la position du joueur (avec un carre 1 par 1)
    def __draw_player_position(self, thickness = 3): # thickness must be odd 
        y, x = self.position
        thickness_range = range(- (thickness // 2), thickness // 2 + 1)
        
        observation_width, observation_height = self.mapshape
        canvas_width, canvas_height, _ = self.canvas_shape

        drawing_width = int(canvas_width / observation_width)
        drawing_height = int(canvas_height / observation_height)
        
        for j in range(y * drawing_height, y * drawing_height + drawing_height):
            try :
                for t in thickness_range:
                    self.canvas[x * drawing_width + t, j] = BLACK
                    self.canvas[(x + 1) * drawing_width + t, j] = BLACK
            except IndexError : pass

        for i in range(x * drawing_width, x * drawing_width + drawing_width):
            try :
                for t in thickness_range:
                    self.canvas[i, y * drawing_height + t] = BLACK
                    self.canvas[i, (y + 1) * drawing_height + t] = BLACK
            except IndexError : pass
            
        pass
    
    # fonction qui dessine tous les elements sur la carte
    def draw_elements_on_canvas(self):
        
        # dessine chaque batiment sur la carte (pour chaque cellule de la carte)
        for y in range(len(self.map)):
            for x in range(len(self.map[0])):
                # couleur de base si aucun batiment n'a ete construit sur cette cellule
                color = WHITE

                # trouve la couleur correspondant au batiment
                if   self.map[y, x] == OFFICE : color = BLUE
                elif self.map[y, x] == HOUSE  : color = RED
                elif self.map[y, x] == PARK  :  color = GREEN
                elif self.map[y, x] == COM  :   color = CYAN
                
                self.__draw_element_on_canvas(y, x, color)
            pass
               
        # dessine la position du joueur et la zone d'observation
        self.__draw_player_position()
        self.__draw_area_position()
        pass
    
    # fonction qui convertit la carte en image
    def to_image(self):
        # chaque cellule de la carte represente un pixel sur la future image
        image = np.ones((self.mapshape[0], self.mapshape[1], 3), dtype = np.uint8) * 0
        
        # pour chaque cellule de la carte
        for y in range(len(self.map)):
            for x in range(len(self.map[0])):
                # couleur de base si aucun batiment n'a ete construit sur cette cellule
                color = WHITE
                
                # trouve la couleur correspondant au batiment
                if   self.map[y, x] == OFFICE : color = BLUE
                elif self.map[y, x] == HOUSE  : color = RED
                elif self.map[y, x] == PARK   : color = GREEN
                elif self.map[y, x] == COM    : color = CYAN
                
                # change la couleur de l'image en fonction de la couleur du batiment
                image[y, x] = color
        # retourne l'image apres edition de tous les pixels
        return image
    
    # fonction de rendu
    def render(self, mode = "console"):
        if mode == "human" :
            # ajoute la valeur de la recompense en haut a droite de l'image
            cv2.putText(self.canvas, str(self.reward), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            # affiche l'image
            cv2.imshow("", self.canvas)
            # attend 1 milliseconde (le temps de bien afficher l'image)
            cv2.waitKey(1)
            # retourne l'image affiche si besoin de l'editer
            return self.canvas
        if mode == "console" :
            # affiche uniquement la position du joueur
            print(self.position)
    
    # fonction qui ferme l'environnement
    def close(self):
        pass
    
if __name__ == "__main__":
    # teste si l'environnement fonctionne sur une ville 20 par 20
    env = City((20, 20),path='./')