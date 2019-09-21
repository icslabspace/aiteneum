# Movies recommender based on collaborative filtering and MovieLens

## Collaborative filtering

```recommender.models.cofi```

One could use the collaborative filtering implementation to produce a features and weights matrices `x` and `theta` used to predict recommendations.

To get this trained model one needs to call `gd-train` function with 4 matrices and two maps:

* 1st arg: x is a MxK real matrix
* 2nd arg: y is a MxN real matrix
* 3rd arg: r is a MxN binary matrix (it matches all zeros in y and puts 1.0s (or 1s) where it doesnt)
* 4th arg: theta is a NxK matrix

* 5th arg: is a map containing 3 functions: `{:keys [rcost-f rgx-f rgtheta-f] :as r-fns}` ; all functions receive the same input: `[x y r theta lambda]`

  * `rcost-f` is a function returning a single number 
  * `rgx-f` is a function returning a matrix with gradients for x (so same shape as x)
  * `rgtheta-f` is a function returning a matrix with gradients for theta (so same shape as theta)

* 6th arg: is a map containing parameters: `{:keys [lambda alpha epsilon no-iters] :as params}` where:

  * `lambda` is the regularization rate
  * `alpha` is the gradient descent update/learning rate
  * `epsilon` is a threshold for stoping the training (before the no-iters is consumed)
  * `no-iters` is the maximum number of iterations that training is allowed to run

```Clojure
(let [x [[1 2] [3 4] [5 6]]
      y [[5] [4] [0]]
      r [[1] [1] [0]]
      theta [[0.14 0.88]]]

    (recommender.models.cofi/gd-train x y r theta
    				      {:rcost-f recommender.models.cofi/regularized-linear-cost
                                       :rgx-f recommender.models.cofi/regularized-gradx 
                                       :rgtheta-f recommender.models.cofi/regularized-gradtheta}
                                       {:lambda 1.5 :alpha 0.001 :epsilon 0.01 :no-iters 10}))				       
```	

This will output:

```Clojure
"Iteration no: " 10 0.14 1
"Iteration no: " 9 0.14307 0.998934
"Iteration no: " 8 0.14604767456065174 0.9978772680027608
"Iteration no: " 7 0.14893645395839913 0.9968294788389025
"Iteration no: " 6 0.1517396364054785 0.9957903230091617
"Iteration no: " 5 0.154460393335085 0.9947595058285035
"Iteration no: " 4 0.1571017744670766 0.9937367466220384
"Iteration no: " 3 0.1596667126638269 0.9927217779693314
"Iteration no: " 2 0.16215802858536812 0.9917143449939317
"Iteration no: " 1 0.1645784351525233 0.9907142046951738
"Iteration no: " 0 0.16693054182631142 0.9897211253195066
"Iterations spent: " 0

[ ;; trained final x (features) matrix
[[0.9897211253195066 1.997474042997839]
  [2.9552414278201855 3.9400620688520833]
  [4.925504230306068 5.910605076367282]]
  ;; trained final theta (weights) matrix
 [[0.16693054182631142 0.92589889709618]]]
 ```


## [MovieLens](https://grouplens.org/datasets/movielens/) datasets

For the CoFi to be of practical use one needs to associate x,y,r and theta with real data sets. As an example the `recommender.data.movie-lens` namespace allows you to build the y and r matrix based on an actual movie list and user ratings for each movie in that list.

The actual [MovieLens](https://grouplens.org/datasets/movielens/) csv files are stored in the `resources/recommender/movie-lens/ml-latest-small/` directory.

To read the movies csv into a useful Clojure movies data structure (movie-list - altough it's a seq not necessarily a list :):

`recommender.data.movie-lens/get-movies`

## Train and save

Usual practical examples mean that matrices x, y and theta are large so training is expensive.

For that reason we can train a model a save it to a file for later use.
To do so one could use the `recommender.train-and-save` namespace.

For example, to save a model tra

```Clojure
(defn comedies-model
  "Train a Collaborative Filtering model (for no-iters iterations) for all comedies in a MovieLens data set and serialize it to disk"
  [no-iters]
  (movies-model movie-lens/movies-csv movie-lens/ratings-csv

                parts/compute-ynorm
                (fn [_] 0.0) ;; parts/compute-ymean 

                (assoc params :genre :comedy :no-iters no-iters)))
```

