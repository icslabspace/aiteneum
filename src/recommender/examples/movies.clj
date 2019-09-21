(ns recommender.examples.movies
  {:author "Alex Gherega" :doc "Putting everything together on how to get movie recommendations starting from the MovieLens dataset"}
    
  (:require [clojure.data.csv :as csv]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as lin]
            [clojure.java.io :as io]
            [fastmath.core :as fmc]
            [fastmath.stats :as fms]
            [recommender.models.cofi :as mc]
            [recommender.data.movie-lens :as mov-l]
            [recommender.data.serializer :as ser]
            [recommender.movies :as movies]
            [vspace.particles :as vsp]))

(m/set-current-implementation :vectorz)

;; ;; BASIC TEST CASE
;; you don;t need this init when using Y-test.csv and R-test.csv are already set with this extracolumn
;; (def my-ratings [[0 4]
;;                  [28 5]
;;                  [42 5]
;;                  [50 5]
;;                  [63 1]
;;                  [73 5]
;;                  [88 5]
;;                  [123 3]
;;                  [163 5]
;;                  [172 5]
;;                  [173 5]
;;                  ;[]
;;                  ])

;; (def y (with-open [reader (io/reader "resources/Y.csv" ;;"resources/Y-test.csv"
;;                                      )]
;;          (doall
;;           (mapv #(mapv read-string %) (csv/read-csv reader)))))

;; (def r (with-open [reader (io/reader "resources/R.csv" ;;"resources/R-test.csv"
;;                                      )]
;;          (doall
;;           (mapv #(mapv read-string %) (csv/read-csv reader)))))

;; (def num-features 10)
;; (def num-users (-> y m/shape second))
;; (def num-movies (-> y m/shape first))


;; (def myr-matrix (vec (repeatedly num-movies #(vector 0))) ;;(m/zero-matrix num-movies 1)
;;   )

;; ;;(doall (pmap (fn [[id r]] (m/mset! myr-matrix id 0 r)) my-ratings))

;; (def y (mapv #(into %1 %2)
;;              (reduce (fn [myrm [id r]] (assoc myrm id [r])) myr-matrix my-ratings)
;;              y))

;; ;;(doall (pmap (fn [[id r]] (m/mset! myr-matrix id 0 1)) my-ratings))
;; (def r (mapv into
;;              (reduce (fn [myrm [id r]] (assoc myrm id [1.0])) myr-matrix my-ratings)
;;              r))

;; (def num-users (-> y m/shape second))
;; (def num-movies (-> y m/shape first))

(def model-file "resources/recommender/comedy-model-[lambda=1.5][alpha=0.001][no-iters=300].edn")

(defn pprint-recommend-movies [search-str]
  (let [comedies-list (mov-l/get-movies mov-l/movies-csv :comedy)
        comedies-model (ser/baby-slurp model-file)


        movies (recommender.data.movie-lens/title->movies comedies-list search-str)
        movie-ids (mapv first movies)]
    
    (clojure.pprint/pprint movies)
    
    (newline)
    (prn "Recommendations for you: ")

    ;; by medoid
    (clojure.pprint/pprint (movies/predict-by-cossim comedies-model
                                                     movie-ids
                                                     comedies-list
                                                     vsp/medoid
                                                     30))

    ;; by centroid
    ;; (clojure.pprint/pprint (movies/predict-by-cossim comedies-model
    ;;                                                  movie-ids
    ;;                                                  comedies-list
    ;;                                                  vsp/centroid
    ;;                                                  30))
    ))
