(ns vspace.particles
  {:author "Alex Gherega" :doc "Some vector space utilities"}
  
  (:require [clojure.spec.alpha :as s]
            [clojure.core.matrix :as m]
            [clojure.spec.test.alpha :as stest]
            [fastmath.vector :as fmv]))
;;-------------------------------------
;; vector space utility functions

(defn centroid [xs]
  (m/scale (apply m/add xs) (/ 1.0 (count xs))))

(defn- codistance [v x xs]
  (into v (mapv #(fmv/dist x %) xs)))

(defn- codistance-matrix [xs]
  (loop [x (first xs)
         coll xs
         v []
         res []]
    (if (-> coll seq not)
      res
      (recur (second coll)
             (rest coll)
             (conj v 0.0)
             (conj res (codistance v x coll))))))

(defn medoid [xs]
  (let [codistances (codistance-matrix xs)
        codistances (m/add codistances (m/transpose codistances))
        sum-codistances (map-indexed #(vector %1 (m/esum %2)) codistances)
        medoid-idx (->> sum-codistances (sort-by second) first first)
        _ (prn "medoid index is: " medoid-idx)]
    (nth xs medoid-idx)))

;; ------------------------------------
;; specs
(s/def ::num-vector (s/coll-of number?))

(s/fdef centroid
  :args (s/and (s/cat :xs (s/coll-of ::num-vector))
               #(let [len (-> % :xs first count)]
                  (every? (fn [v] (-> v count (= len))) (:xs %))))
  :ret vector?
  :fn #(= (-> % :ret count)
          (-> % :args :xs first count)))

(s/fdef medoid
  :args (s/and (s/cat :xs (s/coll-of ::num-vector))
               #(let [len (-> % :xs first count)]
                  (every? (fn [v] (-> v count (= len))) (:xs %))))
  :ret vector?
  :fn #(some (-> % :ret set)
          (-> % :args :xs)))

(stest/instrument [`centroid
                   `medoid])

