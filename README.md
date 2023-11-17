# 8INF804 - TP3
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

    usage: tp3 [-h] [-i INPUT] [-o OUTPUT] [-e EPOCHS] [-b BATCH_SIZE] [-r RUNS] [-l LEARNING_RATE] [-w WORKERS] [-s] {vgg16,custom}

    tp3

    positional arguments:
    {vgg16,custom}                                      neural network to use

    options:
    -h, --help                                          show this help message and exit
    -i INPUT, --input INPUT                             input folder name
    -o OUTPUT, --output OUTPUT                          output folder name
    -e EPOCHS, --epochs EPOCHS                          number of epochs to do
    -b BATCH_SIZE, --batch_size BATCH_SIZE              batch size to use
    -r RUNS, --runs RUNS                                number of runs to do
    -l LEARNING_RATE, --learning_rate LEARNING_RATE     learning rate to use
    -w WORKERS, --workers WORKERS                       number of workers to use
    -s, --show                                          show learning curve graphs

### Exemple d'utilisation

- Les images doivent être placées dans le dossier ./images/ et être au format .png, .jpg ou .jpeg.

- Pour exécuter le programme à partir de la racine du projet, utilisez la commande :

    `python3 tp3 vgg16 ou python3 tp3 custom`

- Les résultats seront enregistrés par défaut dans le dossier ./output.

- Vous pouvez spécifier un autre dossier de sortie avec :

    `python3 tp3 -o ./autre_dossier_sortie vgg16`

- Pour afficher les graphiques des courbes d'apprentissage, utilisez l'option -s :

    `python3 tp3 -s vgg16`

- Pour spécifier le dossier d'entrée, le nombre d'epochs, la taille des batchs, le nombre de runs, le taux d'apprentissage et le nombre de threads, utilisez les options correspondantes :

    `python3 tp3 vgg16 -i ./input/ -e 20 -b 32 -r 3 -l 0.0005 -w 8`

- Par exemple, pour exécuter le réseau de neurones personnalisé avec des paramètres spécifiques :

    `python3 tp3 custom -i ./images/ -o ./output/ -e 15 -b 64 -r 5 -l 0.001 -w 12`