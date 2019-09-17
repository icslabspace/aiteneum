(ns recommender.movies
  {:author "Alex Gherega" :doc "basic API for predicting movie recommendations"}
  
  (:require [clojure.data.csv :as csv]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as lin]
            [clojure.java.io :as io]
            [fastmath.core :as fmc]
            [fastmath.stats :as fms]
            [recommender.models.cofi :as mc]
            [recommender.data.movie-lens :as movie-lens]
            [recommender.data.serializer :as ser]
            [recommender.particles :as parts]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]))

(m/set-current-implementation :vectorz)
;; ---------------------------

(defn predict-by-ratings [[x-final theta-final ymean :as movies-model] n]
  (take n (reverse (sort-by second (map-indexed vector 
                                                (m/add (m/get-column (m/mmul x-final (m/transpose theta-final)) 0)
                                                       (m/get-column ymean 0)))))))

(defn predict-by-cossim [[x theta ymean :as movies-model]
                         movie-ids
                         movies-list
                         projection-fn
                         n]
  (let [num-movies (count movies-list)
        movieid->ymatrixid (movie-lens/nativeids->matrixids movies-list
                                                            num-movies
                                                            first)
        ;; ymatrixid->movieid (movie-lens/matrixids->nativeids movies-list num-movies first) ;; use this for testing
        x-ids (map movieid->ymatrixid movie-ids)
        x-interesting-vector (projection-fn (mapv #(m/get-row x %) x-ids))
        distances (map-indexed
                   (fn [idx x-vector]
                     (vector idx
                             (parts/cos-sim x-interesting-vector
                                            x-vector)))
                   x)]
    
    (mapv #(->> % first (nth movies-list) second)
          (take n (sort-by second distances)))))


;; --------------------------------------
;; specs

;;TODO:
;; (s/fdef predict-by-ratings
;;   :args (s/cat :movies-model))



;; (stest/instrument [`predict-by-ratings
;;                    `predict-by-cossim])
