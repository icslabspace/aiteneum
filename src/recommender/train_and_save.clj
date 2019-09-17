(ns recommender.train-and-save
  (:require [clojure.core.matrix :as m]
            [recommender.data.movie-lens :as movie-lens]
            [recommender.particles :as parts]
            [recommender.models.cofi :as mc]
            [recommender.data.serializer :as ser]))

(def params {:lambda 1.5 :alpha 0.001 :epsilon 0.01 :no-iters 50})

(defn train-model
  ([x [ynorm ymean :as y] r theta r-fns params]
   (let [_ (prn "the shape of theta: " (m/shape theta))
         
         [x-final theta-final] (mc/gd-train x ynorm r theta r-fns params)]

     [x-final theta-final ymean]))
  
  ([movies-list ratings [ynorm ymean :as y] r params]
   (let [num-features movie-lens/num-features
         num-users (-> ynorm m/shape second)
         num-movies (-> ynorm m/shape first)

         x (parts/random-matrix num-movies num-features)
         theta (parts/random-matrix num-users num-features)

         r-fns {:rcost-f recommender.models.cofi/regularized-linear-cost
                :rgx-f recommender.models.cofi/regularized-gradx
                :rgtheta-f recommender.models.cofi/regularized-gradtheta}]
     
     (train-model x y r theta r-fns params))))

(defn- directed-spit [model model-type params]
  (let [path (str "resources/recommender/" (name model-type) "-model-")
        path (reduce #(str %1 "[" (-> %2 key name) "=" (val %2) "]") path
                     params)]
    (ser/baby-spit model #(mapv ser/m->clj %)
                   (str path ".edn"))))

(defn movies-model
  ([movies-csv ratings-csv ynorm-fn ymean-fn params]
   (let [genre (:genre params)
         movies-list (movie-lens/get-movies movies-csv genre)
         ratings (movie-lens/get-ratings ratings-csv movies-list)
         y (m/sparse-matrix (movie-lens/create-ymatrix movies-list ratings))
         r (m/sparse-matrix (movie-lens/create-rmatrix y))
         ymean (ymean-fn y)
         ynorm (ynorm-fn y ymean r)
         movies-model (train-model movies-list ratings [ynorm ymean] r params)]

     (directed-spit movies-model genre (dissoc params :epsilon :genre)))))

(defn comedies-model [no-iters]
  (movies-model movie-lens/movies-csv movie-lens/ratings-csv

                parts/compute-ynorm
                (fn [_] 0.0) ;; parts/compute-ymean 

                (assoc params :genre :comedy :no-iters no-iters)))
