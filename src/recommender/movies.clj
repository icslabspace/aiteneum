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
            [vspace.particles :as vsp]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]))

(m/set-current-implementation :vectorz)
;; ---------------------------

(defn predict-by-ratings
  "Given a [MovieLens] CoFi trained model, for which the first column corresponds to the user's -to which we're recommending- preferences the function outputs a collection of n tuples of structure [movieId, rating] sorted in decreasing order of ratings."
  [[x-final theta-final ymean :as movies-model] n]
  (take n (reverse (sort-by second (map-indexed vector 
                                                (m/add (m/get-column (m/mmul x-final (m/transpose theta-final)) 0)
                                                       (m/get-column ymean 0)))))))

(defn predict-by-cossim
  "Given a [MovieLens] CoFi trained model, a set of movies' ids and a movie list extract the vector representation for the movies' ids set, apply some projection function (e.g. medoid, centroid) which turns a collection of points into a single point. Use the cosine similaity to discover the n closesth vector movies against the projected point. Use the movie list to return actual movies' names for the n discovered points.
  <point and vector mean the same thing in the above description>"
  
  [[x theta ymean :as movies-model]
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

(s/def ::model :recommender.data.serializer/model)
(s/def ::title string?)

(s/fdef predict-by-ratings
  :args (s/and (s/cat :model ::model
                      :n pos-int?)
               #(<= (:n %) (-> % :model :x m/shape first)))
  :ret (s/coll-of (s/tuple nat-int? nat-int?)))


(s/fdef predict-by-cossim
  :args (s/and (s/cat :model ::model
                      :movie-ids (s/coll-of nat-int?)
                      :movies-list :recommender.data.movie-lens/movies-list
                      :projection-fn #{vspace.particles/centroid vspace.particles/medoid}
                      :n pos-int?))
  :ret (s/coll-of ::title :kind vector?))

;; (stest/instrument [`predict-by-ratings
;;                    `predict-by-cossim])
