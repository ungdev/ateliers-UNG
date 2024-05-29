
| ⚙️ / 🧠      | 🏷️ / ❌🏷️       | 🔢 / 🗳️       |
| -------------- | --------------- | --------------- |
| ⚙️ Machine Learning    | 🏷️ Supervised Learning     | 🔢 Régression     |


<br>

## Sommaire
- [👨‍🏫 Présentation de la régression linéaire](#-régression-linéaire)
- [🛠️ Traitement des données](#️-1-traitement-des-données-ou-presque)
- [👷 Pratique de la régression](#-2-pratique-de-la-régression)

<br>

# 👨‍🏫 Régression linéaire

Une régression linéaire, c'est quoi ?
Concrètement, c'est juste une fonction affine *(pour rappel, `f(x) = ax + b`...oui ça remonte à longtemps, on comprend)* qu'on essaie de faire passer au plus proche de nos points.

Par exemple, sur l'image ci-dessous, on considère qu'on veut représenter le prix d'une maison en fonction de sa surface. On a, en vert, des informations du marché (une maison à 5k€ pour 5m2, une maison à 15k€ pour 20m2...), et notre algorithme va tracer une droite qui approxime au mieux toutes ces valeurs (l'approximation est représentée par les points oranges).

<img src=img/linear_reg.png>

<br><br>

<details><summary><b> 💭 Apparté importante : les métriques en IA </b></summary>
<br>
Pour calculer la performance du modèle, parce que c'est quand même mieux de savoir comment son modèle s'en sort, il existe tout un tas de métriques différentes qui veulent toute dire quelque chose de différent.

<br>

Fondamentalement, quand on travaille avec un modèle d'IA, le but final est de chercher à réduire au maximum les différentes métriques d'erreur qu'on peut avoir.

<br>

Pour une régression linéaire, on utilise entre autres le calcul des **moindres carrés** *(MSE : moyenne des résidus au carré, soit la différence entre la valeur réelle et prédite)*.

<img src=img/mse.png width="200">

<br>

Le but, c'est de **minimiser** cette valeur, et donc minimiser la fonction calculant l'erreur du modèle.

> *On utilisera plus tard la ***RMSE***, soit la racine de la MSE.*

L'autre métrique très simple d'utilisation et qu'on utilise partout, c'est *l'accuracy* 🎯, soit la précision du modèle. Tout simplement, c'est la proportion de bonnes prédictions du modèle.


### ➡️ Quelles qu'elles soient, il faut **toujours** évaluer les modèles sur des métriques mathématiques. 

</details>

<br>


## 🛠️ 1. Traitement des données (ou presque)

Pour mettre en pratique une régression linéaire, on utilise un simple jeu de données sur le prix des maisons. Ce jeu de données présente 19 dimensions, mais toutes les variables ne sont pas importantes à conserver, car pas forcément fortement corrélée au prix !
Pour ça, on représente une matrice de corrélation, qui calcule le "score" de corrélation entre chacune des variables (on utilise le calcul de [Pearson](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)).

Concrètement, plus la case est claire, plus les valeurs sont corrélées (positivement, ou négativement). On va donc exclure beaucoup de variables, comme `yr_renovated`, `sqft_lot` ou `condition`. On finit avec un jeu de données à 4 dimensions (pour l'exemple ici).

Si vous voulez traiter le jeu de données vous-même, vous pouvez le récupérer sur Kaggle :
> [House Sales in King County, USA](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction/data)

Sinon, voici le jeu de données pré-traité :
> [SIMPLE_House Sales in King County, USA](kc_house_data.csv)

<br>

## 👷 2. Pratique de la régression 

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
<i>Dans une régression linéaire, ça ne changera rien de scale les données, puisque diminuer l'échelle des données va simplement faire augmenter de manière inversement proportionnelle les coefficients de la régression linéaire, ce qui annulera l'effet du scaler. Mais par principe, on le fait à chaque fois, ça ne coûte rien, et ça peut éviter des problèmes.</i>

<bt/>

La partie de mise en place faite, on s'occupe de l'algorithme.
En intelligence artificielle, il faut toujours **séparer les jeux de données** en deux groupes distincts :
- les données utilisées pour **entraîner** le modèle
- les données utilisées pour **tester** le modèle

Pendant sa phase d'apprentissage, on entraîne le modèle sur généralement **80%** du jeu de données, pour ensuite l'évaluer sur **20%** du jeu de données, qu'il n'a jamais vu auparavant et qu'il n'utilisera pas pour améliorer son modèle. Cela permet de tester la robustesse, capacité à généraliser, du modèle face à des données nouvelles.

<img src=img/test_train.png>

<br>

Pour séparer nos données, on utilise la fonction `test_train_split()` de la librairie `scikit-learn` qui fait tout pour nous, en utilisant une *seed* qui nous permet d'avoir un entraînement déterministe (*résultat aléatoire* = f(seed)) :
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

> 🐈‍⬛ Le code final est disponible dans ce dossier, sur le GitHub.

## Bravo ! 🎉
Vous êtes arrivés à la fin de cette partie et avez rédigé votre premier algorithme de Machine Learning 👏.

Vous êtes maintenant libre de suivre la suite des chapitres de l'atelier, que ce soit en Machine Learning ou Deep Learning, en vous référant au [sommaire](/README.md#-sommaire-des-ateliers)

Nous restons à votre disposition pour la moindre question, donc n'hésitez pas 😉