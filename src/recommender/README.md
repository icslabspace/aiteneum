# Movies recommender based on collaborative filtering and MovieLens

## Collaborative filtering

```Clojure
(in-ns 'recommender.models.cofi)
```

One could use the collaborative filtering implementation to produce a features and weights matrices `x` and `theta` used to predict recommendations.

To get this trained model one needs to call `gd-train` function with 4 matrices and two maps:

* 1st arg: `x` is a `MxK` real matrix
* 2nd arg: `y` is a `MxN` real matrix
* 3rd arg: `r` is a `MxN` binary matrix (it matches all zeros in `y` and puts 1.0s (or 1s) where it doesnt)
* 4th arg: `theta` is a `NxK` matrix

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

```Clojure
(in-ns 'recommender.data.movie-lens)
```

For the CoFi to be of practical use one needs to associate x,y,r and theta with real data sets. As an example the `recommender.data.movie-lens` namespace allows you to build the y and r matrix based on an actual movie list and user ratings for each movie in that list.

The actual [MovieLens](https://grouplens.org/datasets/movielens/) csv files are stored in the `resources/recommender/movie-lens/ml-latest-small/` directory.

To read the movies csv into a useful Clojure movies data structure:

```Clojure
(def movies (recommender.data.movie-lens/get-movies "resources/recommender/movie-lens/ml-latest-small/movies.csv" :comedy))
```

where the first argument is a string filepath to a MovieLens movies.csv file and the second it one of the following keywords:
```Clojure
[:film-noir
 :western
 :fantasy
 :children
 :animation
 :horror
 :mystery
 :musical
 :romance
 :war
 :drama
 :sci-fi
 :documentary
 :adventure
 :comedy
 :thriller
 :action
 :crime]
 ```
To read the ratings csv into a useful Clojure ratings data structure (i.e. a collection of tuples `[[userId movieId] rating]`:

```Clojure
(recommender.data.movie-lens/get-ratings "resources/recommender/movie-lens/ml-latest-small/ratings.csv" movies)
```
This namespace provides you with the `create-ymatrix` and `create-rmatrix` functions which will output a `y` matrix of `[number-of-movies X number-of-users]` and an `r` matrix of the same shape as `y` but with zeros wherever `y` is zero and with ones wherever `y` is not zero.

The `(clojure.core.matrix/get-element y i j)` will give you the rating for movieId i given by userId j.

## Train and save

```Clojure
(in-ns 'recommender.train-and-save)
```

Usual practical examples mean that matrices `x`, `y` and `theta` are large so training is expensive.

For that reason we can train a model a save it to a file for later use.
To do so one could use the `recommender.train-and-save` namespace.

To train and save a movies model one needs only call the following function:

```Clojure
(movies-model "resources/recommender/movie-lens/ml-latest-small/movies.csv"
	      "resources/recommender/movie-lens/ml-latest-small/ratings.csv"

                recommender.particles/compute-ynorm
                (fn [_] 0.0) ;;or use parts/compute-ymean 

                (assoc params :genre :crime :no-iters no-iters))
```
The `movies-model` function takes in the string filepaths to the `movies.csv` and `ratings.csv`, two functions `ynorm-fn` and `ymean-fn` and a parameters map `{:lambda 1.5 :alpha 0.001 :epsilon 0.01 :no-iters 50 :genre :crime}`.

The `ynorm-fn` is a function of one argument, namely the matrix `y`, and outputs the normalized version of that matrix (e.g. `compute-ynorm` divides each element on a row of `y` by that row's by arithmetic mean value).

The `ymean-fn` is a function of one argument which can output either a number or a column matrix. The column-matrix would store the mean values computed for each row in `y`.

Here is an example on how to train and save a model trained for `no-iters` number of iterations, on all comedies found in the MovieLens `movies.csv` with parameters:

```Clojure
{:lambda 1.5 :alpha 0.001 :epsilon 0.01 :no-iters 5}
```

one could call the `recommender.train-and-save/comedies-model` function which is defined as follows:

```Clojure
(defn comedies-model
  "Train a Collaborative Filtering model (for no-iters iterations) for all comedies in a MovieLens data set and serialize it to disk"
  [no-iters]
  (movies-model movie-lens/movies-csv movie-lens/ratings-csv

                parts/compute-ynorm
                (fn [_] 0.0) ;; parts/compute-ymean 

                (assoc params :genre :comedy :no-iters no-iters)))
```

The above function is already implemented in the `recommender.train-and-save` namespace. You'll find the result stored in the:

`resources/recommender/comedy-model-[lambda=1.5][alpha=0.001][epsilon=0.1][no-iters=5].edn`.

Whenever you use `movies-model` function you can expect a file to be saved in `resources/recommender` using the naming convention:

`<genre>-model-[lambda=<>][alpha=<>][epsilon=<>][no-iters=<>].edn`


## Predict recommendations

```Clojure
(in-ns 'recommender.movies)
```

This namespace exposes two functions: `predict-by-ratings` and `predit-by-cossim`.

The first function requires one to modify the `y` and `r` matrices by adding a new first column containing new movie ratings.

For example:

```Clojure
(def initial-y ;; 3 movies and 5 users
     (clojure.core.matrix/sparse-matrix
     [[5.0 4.5 0.0 0.0 5.0] ;; row corresponding to user ratings for movie "Soul Kitchen (2009)"
      [[5.0 5.0 4.5 0.0 0.0]] ;; row corresponding to user ratings for movie "The Hitchhiker's Guide to the Galaxy (2005)"
      [0.0 0.0 0.0 5.0 5.0] ;; row corresponding to user ratings for movie "Un peu, beaucoup, aveuglément (2015)"
      ]))

(def initial-r
     (clojure.core.matrix/sparse-matrix
	(recommender.data.movie-lense/create-rmatrix initial-y)))
```
*NOTE*: always use sparse-matrix as inputs to `recommender.train-and-save/train-model` or equivalent functions.

Suppose I have my (or a new user has its) own rating 5 star for the "Soul Kitchen" movie. Then one should update the `y` matrix so that is now looks like this:

```Clojure
[[5.0 5.0 4.5 0.0 0.0 5.0] ;; row now includes the 6th user on the first position
 [0.0 5.0 5.0 4.5 0.0 0.0]
 [0.0 0.0 0.0 0.0 5.0 5.0]]
 ```
Similarly the `r` matrix should be updated (just call the `create-rmatrix` on the new `y`). (you should just call `create-rmatrix` once you have your final `y`).

You can now call `recommender.movies/predict-by-ratings` with a CoFi model trained on `y` and `r` and specify the last argument as the number of recommendetions you'd like to get. This function multiplies `x` by `theta` and the first column gives you the predictions.

Alternatively, you can use just the features matrix `x` - which might be a vector space for the movies used to train your model. Each row in the trained `x` is a point and `recommender.movies/predict-by-cossim` function returns the first `n` similar points by cosine similarity. A case example is provided in the next section.

## Examples of getting movies recommendations

Suppose you want to use a saved model:

```Clojure
(def model-file "resources/recommender/comedy-model-[lambda=1.5][alpha=0.001][no-iters=300].edn")
```

which is stored in the `resources/recommender/` directory.

If you want to get the first 30 similar movies based on a search string this is a code you could use:

```
(defn pprint-recommend-movies [search-str]
  (let [;; get the clojure comedies collection 
        comedies-list (mov-l/get-movies mov-l/movies-csv :comedy) ;; this is a collection of tuples [movieId title genre-mix]

	;; load the serialized model
        comedies-model (ser/baby-slurp model-file) ;; this is a tuple holding the features matrix x, the weights matrix theta and the ymean value/collumn-matrix


	;; extract a subcollecion of all movies that have search-str in their title
	movies (recommender.data.movie-lens/title->movies comedies-list search-str)

	;; get the more useful movies' ids
        movie-ids (mapv first movies)]

    ;; display what movies we get for the search-str argument
    (clojure.pprint/pprint movies)
    
    (newline)
    (prn "Recommendations for you: ")

    ;; Using the medoid as a projection function, lookup the closesth 30 points and
    ;; output the movies' titles associated with those points
    (clojure.pprint/pprint (movies/predict-by-cossim comedies-model
                                                     movie-ids
                                                     comedies-list
                                                     vsp/medoid
                                                     30))
    ;; Using the centroid as projectoin functions, lookup the closesth 30 points and
    ;; output the movies' titles associated with those points 
    (clojure.pprint/pprint (movies/predict-by-cossim comedies-model
                                                     movie-ids
                                                     comedies-list
                                                     vsp/centroid
                                                     30)) 
     ))
						     
						     
```


## License

Copyright © 2019 Alexandru Gherega

Distributed under the Eclipse Public License either version 1.0 or (at your option) any later version.

