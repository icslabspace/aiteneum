(ns recommender.particles
  {:author "Alex Gherega" :doc "Some vector space utilities"}
  
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as lin]
            [fastmath.stats :as fms]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]))

;; TODO: move this to vspace/particles.clj
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

;; ----------------------------------
;; specs
(s/def ::num-vector
  (s/and m/vec?
         #(every? number? %)))

(s/fdef random-matrix
  :args (s/cat :n pos-int? :m pos-int?)
  
  :ret m/matrix?

  :fn #(= [(-> % :args :n) (-> % :args :m)]
          (m/shape (-> % :ret))))

(s/fdef compute-ymean
  :args (s/cat :y m/matrix?)
  :ret m/column-matrix?
  :fn #(= (-> % :args :y m/shape first)
          (-> % :ret m/shape first)))


(s/fdef compute-ynorm
  :args (s/and (s/cat :y m/matrix? :ymean (s/or :col-mat m/column-matrix?
                                                :number number?)
                      :r m/matrix)
               #(m/same-shape? (-> % :y) (-> % :r))
               #(if-let [cols (-> % :ymean :col-mat)]
                  (=  (-> cols m/shape first) (-> % :y m/shape first))
                  true))
  :ret m/matrix?
  :fn #(m/same-shape? (-> % :args :y)
                      (-> % :ret)))


(s/fdef cos-sim
  :args (s/cat :v1 ::num-vector
               :v2 ::num-vector)
  :ret (s/and #(<= % 1.0) #(>= % -1.0)))

(stest/instrument [`random-matrix
                   `compute-ymean
                   `compute-ynorm
                   `cos-sim])
