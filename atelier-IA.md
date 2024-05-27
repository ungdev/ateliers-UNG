# 1. Régression linéaire

Une régression linéaire, c'est quoi ?
Concrètement, c'est juste une fonction affine (f(x) = ax + b) qu'on essaie de faire passer au plus proche de nos points.

Par exemple, sur l'image ci-dessous, on considère qu'on veut représenter le prix d'une maison en fonction de sa surface. On a, en vert, des informations du marché (une maison à 5k€ pour 5m2, une maison à 15k€ pour 20m2...), et notre algorithme va tracer une droite qui approxime au mieux toutes ces valeurs.

<img src=img/linear_reg.png>

Pour calculer la performance du modèle, on utilise le calcul des **moindres carrés** (somme des résidus au carré, soit la différence entre la valeur réelle et prédite). Le but, c'est de **minimiser** cette valeur, et donc minimiser la fonction calculant l'erreur du modèle.
Fondamentalement, quand on travaille avec un modèle d'IA, le but final est de chercher à réduire au maximum les différentes métriques d'erreur qu'on peut avoir : les moindres carrés représentent une métrique parmi tant d'autres.


## 1.1 Traitement des données (ou presque)

Pour mettre en pratique une régression linéaire, on utilise un simple jeu de données sur le prix des maisons. Ce jeu de données présente 19 dimensions, mais toutes les variables ne sont pas importantes à conserver, car pas forcément fortement corrélée au prix !
Pour ça, on représente une matrice de corrélation, qui calcule le "score" de corrélation entre chacune des variables (on utilise le calcul de [Pearson](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)).

Concrètement, plus la case est claire, plus les valeurs sont corrélées (positivement, ou négativement). On va donc exclure beaucoup de variables, comme `yr_renovated`, `sqft_lot` ou `condition`. On finit avec un jeu de données à 4 dimensions (pour l'exemple ici).

Si vous voulez traiter le jeu de données vous-même, vous pouvez le récupérer sur Kaggle :
//todo: lien kaggle

Sinon, voici le jeu de données pré-traité :
//todo: lien jeu de données

## 1.2 Pratique de la régression 

Pour la régression linéaire, il faut tout d'abord installer les librairies nécessaires. Pour ce faire, exécutez la commande suivante :

```
pip install numpy scikit-learn pandas matplotlib
```

On va ensuite, dans un fichier Python, importer le jeu de données. Mettez votre `kc_house_data.csv` dans le même dossier que votre fichier Python, qu'on commencera avec le code ci-dessous :

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('kc_house_data.csv')
```

Maintenant vous allez vite remarquer un problème dans nos données : un nombre de chambres, ça oscille entre 1 et 10 disons...alors que les *pieds* carrés, ça oscille entre 200 et 4 000. Quand on calcule une fonction, avoir des ordres de grandeur si variés, ça n'aide pas !
Pour remédier à ça, on utilise un `Scaler` qui va nous faire la mise à l'échelle des valeurs tout seul, *c'est-y-pas magique* ✨.

Pour ce faire, on utilise ce code :
```py
y = data['price']   # y has the price, the data we try to predict
X = data[['bedrooms', 'bathrooms', 'sqft_living']]   # X has the data from which we try to predict the y data (price in this case)

scale = StandardScaler()   # Initialize the scaler
scale.fit(X)   # Fit it to the data
scaled_X = scale.transform(X)   # Transform the data according to the fitted scaler
```


La partie de mise en place faite, on s'occupe de l'algorithme.
En intelligence artificielle, il faut toujours séparer les jeux de données en deux groupes distincts : les données utilisées pour entraîner le modèle, et les données utilisées pour tester le modèle : pendant sa phase d'apprentissage, on entraîne le modèle sur généralement **80%** du jeu de données, pour ensuite l'évaluer sur **20%** du jeu de données, qu'il n'a jamais vu auparavant et qu'il n'utilisera pas pour améliorer son modèle. Cela permet de tester la robustesse, capacité à généraliser, du modèle face à des données nouvelles.

Pour séparer nos données, on utilise la fonction `test_train_split` de la librairie `scikit-learn` qui fait tout pour nous, en utilisant une seed :
```py
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # test_size = proportion of dataset used for testing ; random_state = seed
```

Ceci étant fait, on peut désormais initiliaser le modèle et l'entraîner... et pour une régession linéaire, pas de panique, on a encore des librairies pour nous mâcher le travail !
```py
# Instantiate a linear regression model
model = LinearRegression()

# This step lets the code run on its own all the training process. Right after, 'model' can be called anywhere as a pre-trained model
model.fit(X_train, y_train)
```

A présent... à vous de jouer 🫵. Complétez cette ligne en remplaçant l'intérieur des arguments de `model.predict()`.
```py
# Predict using the trained model
y_pred = model.predict(///)
```

<details><summary>💡 Indice :</summary>
On cherche ici à stocker les prédictions du modèle calculées sur les données de <strong>test</strong>. Donc, on fait une prédiction sur <code>X_test</code> (et non pas <code>y_test</code>, car c'est la "réponse" à ce qu'on cherche : en donnant <code>X_test</code>, on approxime <code>y_test</code> et on rentre ces prédictions dans <code>y_pred</code>).

On a donc :
```py
y_pred = model.predict(X_test)
```
</details>

Etape importante ! Il faut maintenant calculer une métrique d'erreur, de performance, du modèle. Comme présenté en <a href='#1-régression-linéaire'>#1. Régression linéaire</a>, on va ici utiliser la méthode des moindres carrés (et plus précisément, sa racine carrée) pour estimer l'erreur du modèle :
```py
# Calculate the root mean squared error
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print('Root Mean Squared Error:', rmse)
```
On obtient normalement une RMSE d'environ **272 466**. La RMSE dépendra toujours des données utilisées (ici, on travaille sur des prix avec de très grands ordres de grandeur).
Spoiler Alert : c'est quand même pas très bon comme score. Mais ça s'explique par plusieurs facteurs :
- on utilise que 3 dimensions. C'est peu, très peu.
- la relation n'est pas forcément linéaire
- le jeu de données est biaisé (la répartition des prix des maisons est inégale)

Ca n'empêche qu'on peut tout de même représenter visuellement la performance de notre modèle. On a décidé de le représenter autour d'une ligne droite : plus les points sont proches de la ligne rouge (X = Y), plus les prédictions sont bonnes.
```py
# Plot the predicted values against the actual values using a linear regression model
plt.scatter(y_pred, y_test)
# Plot a line x = y
plt.plot([0, max(max(y_test), max(y_pred))], [0, max(max(y_test), max(y_pred))], color='red')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Predicted Price vs Actual Price')
plt.show()
```

Si vous le souhaitez, pour améliorer les performances du modèle, vous pouvez tester avec + de variables. On a ainsi rapidement eu 22% d'erreur en moins...
Le code final est disponible sur le GitHub.


# 2. Réseau de neurones
 
Bouh ça fait peur, mais vous inquiétez pas, c'est en fait assez chill (ou presque).
Construire un réseau de neurones, ça consiste à concaténer tout un tas de fonctions pour avoir des prédictions vraiment meilleures. Comme si on passait notre donnée dans une suite de régressions linéaires (on fait pas ça avec l'algorithme précédent de régression linéaire parce que c'est pas opti et que ça marche juste pas, mais ça vous donne une idée du principe).

Pour ça, on utilise des neurones qui sont disposées sur des couches différentes.

<img src=img/cat_pred.png>

- Sur la **couche d'entrée** (input layer), on va donner au réseau une décomposition de notre donnée d'entrée, si nécessaire. Si on veut prédire l'objet sur une image, on lui passe une image, qui va être décomposée en niveaux de rouge, de bleu, de vert... par exemple, et chacune de ces informations sera traitée par un neurone d'entrée différent.
- Ce sont les **couches cachées** (hidden layers) qui vont faire 99,9% du travail. Ce sont ces neurones-là qui vont faire les calculs et ajouter des couches de complexité aux modèles. Il peut y avoir une, deux, dix couches de neurones cachées, en fonction des performances recherchées.
- Enfin, la **couche de sortie** (output layer) va simplement renvoyer le résultat trouvé par le réseau. Dans l'exemple ci-dessus, le réseau détecte à 62% que l'image est un chien et 38% que c'est un chat, donc le réseau de neurones dira que c'est un chien (dommage).

###### Mais dis-moi Jamy, qu'est-ce qu'un neurone ?
Un neurone, c'est une fonction linéaire qui va faire un simple calcul sur nos données. On va l'accoler à une fonction d'activation, qui va casser la linéarité du réseau (sinon, il servirait pas à grand chose) et donner une sortie précise à notre neurone. Cette sortie est ensuite utilisée par d'autres neurones de la couche suivante, et ainsi de suite. C'est le "maillage" que vous voyez sur l'image : ici, tous les neurones d'une couche sont tous liés à ceux de la couche suivante.

###### Mais dis-moi Jamy, qu'est-ce qu'une fonction d'activation ?
Il en existe tout un tas, mais voici quelques exemples :

<img src=img/activation_functions.png>

Vous retrouvez entre autres, la fonction `seuil` (Perceptron), `ReLU`, qui est une sorte de seuil, la fonction `tangente hyperbolique`, `sigmoïde`...


Là tout de suite, ca fait pas mal de concepts abstraits, mais dites-vous juste qu'un réseau de neurones, ça se contente de :
-> récupérer des données d'entrée
-> traiter ces données avec une myriade de calculs simples
-> donner une sortie qui évalue le degré de confiance du réseau par rapport à sa réponse

Si on simplifie, c'est juste une très très grosse fonction mathématique avec des paramètres à trouver.
Ces paramètres justement, c'est ce qu'on appelle des **poids** et des **biais**. Sur les liens qu'il existe entre les neurones (tous les traits qui connectent les neurones entre eux sur le schéma), on peut coefficienter les valeurs, par exemple, un poids de 0.1 sur un lien va diminuer grandement l'impact du résultat du neurone précédent dans le calcul du neurone suivant.

<img src=img/cat_pred_weights.png>

Sur le neurone B1, on va effectuer un calcul en prenant `0.7*(A1)`, `0.4*(A2)`, `-1.3*(A3)` et `1.4*(A4)` comme entrée.
Par exemple, le neurone B1 peut sommer les entrées, et ajouter un biais, soit : `B1 = 0.7*(A1) + 0.4*(A2) + -1.3*(A3) + 1.4*(A4) + biais`. On utilise une fonction d'activation sur ce neurone, et paf, on a le résultat à envoyer aux prochains neurones.

Si vous n'avez pas tout saisi, ne vous en faites pas, voici un simple exercice pour mieux comprendre comment ça fonctionne. Calculez la sortie du neurone vert.

<img src=img/nn_exercise.png>

<details><summary>💡 Réponse :</summary>
La réponse est <strong>0</strong> !
La somme des 3 neurones bleus, avec les poids respectifs, donne -4.3. Avec le biais, on a -1.3. Avec la fonction d'activation, on a bien max(0, -1.3) = 0.
</details>

<br>

Bon tout ça c'est sympa...mais fort heureusement, vu qu'ici on parle d'intelligence artificielle, c'est la machine qui va faire tous ces calculs, décider des poids à mettre, ou de certaines fonctions à utiliser ! Dans le jargon, on appelle ce procédé de calcul de l'ordinateur de la **forward propagation** ou du **feed-forward**.

###### Mais dis-moi Jamy, comment est-ce qu'on fait pour déterminer les paramètres ?
On ne va pas rentrer dans les détails mathématiques, mais globalement, le but de l'algorithme va être de minimiser la **fonction de coût** associée au réseau. La fonction de coût prend en paramètres les poids et biais, afin d'en sortir une grosse fonction dont il faut trouver le minimum. C'est comme lâcher une bille, et trouver le creux le plus bas.

<img src=img/cost_function_3d.png>

Pour trouver ce minimum, on utilise un procédé mathématique un peu long et pénible qui s'appelle la **descente de gradient**. Concrètement, on calcule itérativement le gradient *(des dérivées de fonctions à plusieurs variables, si vous avez pas encore fait MT04 ou PHYS11)* en descendant la pente jusqu'à trouver le minimum. S'il est local c'est bien, global c'est mieux, mais c'est pas toujours évident (Eviden).

<img src=img/gradient_descent.png>

Beaucoup beaucoup de blabla, mais maintenant, on peut passer à la pratique !

## 2.1 Pratique du réseau de neurones guidée

Nous n'allons pas expliquer étape par étape l'intégration du réseau de neurones. Toutefois, on vous donne l'entièreté du code commenté, pour que vous compreniez, le fonctionnement pas-à-pas du réseau ! Après cela, vous pourrez appliquer un réseau de neurones à un autre jeu de données, sur des images... Donc restez attentifs 😉

```py
# Libraries import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, ELU
from keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('kc_house_data.csv')

X = data.drop(['price'], axis=1)   # We use every feature, except the price we're trying to predict
y = data['price']

# Standardize the data
scale = StandardScaler()   # Initialize the scaler
scale.fit(X)   # Fit it to the data
scaled_X = scale.transform(X)   # Transform the data according to the fitted scaler

# Separate the data as test and train data
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()   # Initialize the structure of the neural network
model.add(Dense(10, input_dim=X_train.shape[1]))   # Add a layer of 10 neurons, the input layer is implicit with X_train.shape[1] neurons
model.add(LeakyReLU(alpha=0.1))   # Tell the model that those first 10 neurons are of activation function LeakyReLU. alpha = slope for negative values (see LeakyReLU graph)
model.add(Dense(32))   # Add a layer of 32 neurons, input shape is inferred from last layer
model.add(ELU(alpha=1.0))   # Add the ELU activation function for those 32 neurons
model.add(Dense(64))   # Add a layer of 64 neurons
model.add(LeakyReLU(alpha=0.1))   # Add the LeakyReLU activation function for those 64 neurons
model.add(Dense(1, activation='linear'))   # Use a linear activation function (= no activation) for output layer. 1 neuron since we want 1 value

optimizer = Adam(learning_rate=0.003)   # Optimizer defines the process to diminish the cost function. Here, we use the Adam optimizer, an already existing algorithm

# Compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predict house prices
y_pred = model.predict(X_test)[:, 0]

rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print('Root Mean Squared Error:', rmse)

# Plot the predicted values against the actual values using a linear regression model
plt.scatter(y_pred, y_test)
# Plot a line x = y
plt.plot([0, max(max(y_test), max(y_pred))], [0, max(max(y_test), max(y_pred))], color='red')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Predicted Price vs Actual Price')
plt.show()
```

## 2.2 Pratique du réseau de neurones autonome

On va maintenant vous laisser pratiquer de vous-même les réseaux de neurones, en repartant de la base précédente, pour travailler sur le jeu de données MNIST, qui contient 70 000 images de chiffres écrits à la main sur des images de 28x28px.

<img src=img/MNIST.jpg>

