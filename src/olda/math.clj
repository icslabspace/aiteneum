(ns olda.math
  (:require [fastmath.core :refer [digamma exp]]
            [clojure.walk :refer [postwalk]]
            [clojure.core.matrix :refer [esum emap add sub transpose vec? array?] :as m]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]))

(defn psi
  "Compute the psi function for a matrix input"
  [matrix]
  (postwalk #(if (number? %)
               (digamma %)
               %)
            matrix))

(defn psi-of-sum
  "Add columns of a matrix and then compute
  the psi function on that vector"
  [matrix]
  (->> matrix
       ;; transpose and add is equivalent to: (into [] (map #(apply + %)) alpha)
       transpose
       (apply add)
       psi))

(defn expn
  "Compute e^(ai) where ai are inputs of a 2D matrix;
  it returns a 2D matrix"
  [array]
  (emap #(exp %) array))

(defn substitute
  "Substitute a vector into another"
  [v1 v2]
  (let [size1 (count v1)
        size2 (count v2)]
    (apply conj v1 (subvec v2 size1 size2))))

(defn replace-block
  "Given a block matrix and a larger matrix m replace
  block into m starting at position <0,0> in m"
  [block m]
  ;; block shape is always smaller than m's
  (let [get-row #(into [] (get %1 %2))]
    (into [] (map-indexed #(substitute (get-row block %1) %2)) m)))

(defn reshape-strict
  "Replace a block constructed given a specific shape into
  a matrix m statring at position <0,0> in m;
  the block will be zero populated unless a scalar value is passed as the third argument"
  [m shape & scalar] ;; shape is always at least as large as m's
  (let [val (or (first scalar) 0)
        nm (apply m/new-matrix shape)]
    (replace-block m (-> nm (m/fill val) m/to-nested-vectors))))

(defn normalize
  "Given a collection of number values do a basic normalization"
  [v]
  (let [sv (apply + v)]
    (into [] (map #(/ % sv)) v)))

;; spec

(s/def ::number-array (s/and array? #(->> % flatten (every? number?))))

(s/fdef psi
        :args (s/cat :matrix-or-number (s/or :matrix ::number-array
                                             :number number?))
        :ret ::number-array)

(s/fdef psi-of-sum
        :args (s/cat :alpha ::number-array)
        :ret ::number-array)

(s/fdef expn
        :args (s/cat :array ::number-array)
        :ret ::number-array)

(stest/instrument [`psi `psi-of-sum `expn])

