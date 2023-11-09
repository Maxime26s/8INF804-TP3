# 8INF804 - TP2
## Auteurs
- Maya Legris (LEGM15600100)
- Tifenn Le Gourriérec (LEGT08590100)
- Maxime Simard (SIMM26050001)

## Prérequis
Python >= 3.9

## Installation

### GitHub

- Cloner le *repository* du projet

### Moodle

- Décompresser le fichier *zip* du projet

## Utilisation

- Toutes les commandes sont à exécuter à partir de la racine du projet

### Environnement virtuel

- Afin de facilité l'utilisation du projet, il est recommandé d'utiliser un environnement virtuel
- Pour se faire, exécuter la commande suivante:

    `python3 -m venv venv`

- Activer l'environnement virtuel sur Windows:

    `.\venv\Scripts\activate`

- Pour installer les dépendences, exécuter la commande suivante:

    `pip install -r requirements.txt`

### Arguments

    usage: tp2 [-h] [-o OUTPUT] [-s] [-sc SCALE] input

    tp2

    positional arguments:
    input                 input folder name

    options:
    -h, --help            show this help message and exit
    -o, --output OUTPUT   output folder name, default is ./output/
    -s, --show            show processed images
    -a, --algorithm ALGORITHM  algorithm to use, default is 'all'
    -sc, --scale SCALE    scale images, default scale is 1

### Exemple d'utilisation

- Le dossier d'images doit se trouver à l'emplacement ./images/ et contenir des fichiers au format .png, .jpg ou .jpeg

- À partir de la racine du projet, exécuter la commande suivante:

    `python3 tp2 ./images/`

- Par défaut, les résultats sont enregistrés dans le dossier ./output

- Il est aussi possible de spécifier un autre dossier de sortie:

    `python3 tp2 -o ./out ./images/`

- Pour afficher les images traitées, utiliser l'option -s:

    `python3 tp2 -s ./images/`

- Pour sélectionner un algorithme spécifique pour le traitement, utiliser l'option -a suivie du numéro de l'algorithme [1, 4] (par exemple, '1' pour segment_1) :

    `python3 tp2 -a 1 ./images/`

- Pour sélectionner tous les algorithmes, utiliser l'option -a de "all" :

    `python3 tp2 -a all ./images/`

- Pour changer l'échelle des images, utiliser l'option -sc avec un facteur d'échelle (ex: 0.5 pour réduire de moitié):

    `python3 tp2 -sc 0.5 ./images/`

- Ainsi, les images seront traitées en utilisant l'échelle spécifiée.