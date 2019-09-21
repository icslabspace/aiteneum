(ns recommender.train-and-save
  {:author "Alex Gherega" :doc "Build pre-trained models for movie recommendations and store them on disk"}
  
  (:require [clojure.core.matrix :as m]
            [recommender.data.movie-lens :as movie-lens]
            [recommender.particles :as parts]
            [recommender.models.cofi :as mc]
            [recommender.data.serializer :as ser]
            [clojure.spec.alpha :as s]))

(def params {:lambda 1.5 :alpha 0.001 :epsilon 0.01 :no-iters 50})

(defn train-model
  "Closure over mc/gd-train. Prepares all necessary inputs to run gradient descent"
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

(defn- directed-spit
  "File name formatter and saver."
  [model model-type params]
  (let [path (str "resources/recommender/" (name model-type) "-model-")
        path (reduce #(str %1 "[" (-> %2 key name) "=" (val %2) "]") path
                     params)]
    (ser/baby-spit model #(mapv ser/m->clj %)
                   (str path ".edn"))))

(defn movies-model
  "Build a movies models based on certain parasm (e.g. for all comedy movies)"
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

(defn comedies-model
  "Train a Collaborative Filtering model (for no-iters iterations) for all comedies in a MovieLens data set and serialize it to disk"
  [no-iters]
  (movies-model movie-lens/movies-csv movie-lens/ratings-csv

                parts/compute-ynorm
                (fn [_] 0.0) ;; parts/compute-ymean 

                (assoc params :genre :comedy :no-iters no-iters)))


;; ----------------------------------------
;; specs

(s/def ::ymean :recommender.data.serializer/ymean)

(s/def ::ynorm-ymean (s/tuple m/matrix? ::ymean))

(s/def ::model :recommender.data.serializer/model)

(s/fdef train-model
  :args (s/or :f1 (s/and (s/cat :x m/matrix?
                                :y ::ynorm-ymean
                                :r :recommender.particles/binary-matrix
                                :theta m/matrix?
                                :r-fns :recommender.models.cofi/r-fns
                                :params :recommender.models.cofi/params)
                         
                         #(m/same-shape? (-> % :y first) (:r %))
                         
                         #(= (-> % :x m/shape) [(-> % :y first m/shape first)
                                                (-> % :theta m/shape second)])
                         
                         #(= (-> % :theta m/shape) [(-> % :y first m/shape second)
                                                    (-> % :x m/shape second)]))
              :f2 (s/and (s/cat :movies-list :recommender.data.movie-lens/csv-data
                                :ratings :recommender.data.movie-lens/ratings
                                :y ::ynorm-ymean
                                :r :recommender.models.cofi/binary-matrix
                                :params :recommender.models.cofi/params)))
  :ret ::model)

(s/fdef directed-spit
  :args (s/cat :model ::model
               :model-type string?
               :params map?))

(s/fdef movies-model
  :args (s/cat :movies-csv string? ;;:recommender.data.movie-lens/csv-data
               :ratings-csv string? ;;:recommender.data.movie-lens/csv-data
               :ynorm-fn ifn?
               :ymean-fn ifn?
               :params :recommender.models.cofi/params))

(s/fdef comedies-model
  :args (s/cat :no-iters pos-int?))

