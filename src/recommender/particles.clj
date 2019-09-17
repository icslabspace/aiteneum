(ns recommender.particles
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as lin]
            [fastmath.stats :as fms]))

(defn random-matrix [n m]
  (m/matrix 
   (repeatedly n
               (fn [] (repeatedly m
                                  #(* (rand-nth [-1 1]) (rand)))))))

(defn compute-ymean [y]
  (m/column-matrix (mapv (fn [y-row] (fms/mean (filter #(-> % zero? not)
                                                       y-row)))
                         y)))

(defn compute-ynorm [y ymean r]
  (if (m/matrix? ymean)
    (m/sub y (mapv #(m/scale %1 (first %2)) r ymean))
    (m/sub y (mapv #(m/scale %1 (first %2)) r (m/add (->> y
                                                          m/shape
                                                          (apply m/zero-matrix))
                                                     ymean)))))


(defn cos-sim [v1 v2]
  (/ (m/dot v1 v2)
     (* (lin/norm v1) (lin/norm v2))))
