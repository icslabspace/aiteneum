(ns olda.math
  (:require [fastmath.core :refer [digamma exp]]
            [clojure.walk :refer [postwalk]]
            [clojure.core.matrix :refer [esum emap add sub transpose vec? array?] :as m]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]))

(defn psi [matrix]
  (postwalk #(if (number? %)
               (digamma %)
               %)
            matrix))

(defn psi-of-sum [alpha]
  (->> alpha
       ;; transpose and add is equivalent to: (into [] (map #(apply + %)) alpha)
       transpose
       (apply add)
       psi))

(defn expn [array]
  (emap #(exp %) array))

(defn substitute [v1 v2]
  (let [size1 (count v1)
        size2 (count v2)]
    (apply conj v1 (subvec v2 size1 size2))))

(defn replace-block [block m] ;; block shape is always smaller than m's
  (let [get-row #(into [] (get %1 %2))]
    (into [] (map-indexed #(substitute (get-row block %1) %2)) m)))

(defn reshape-strict [m shape & scalar] ;; shape is always at least as large as m's
  (let [val (or (first scalar) 0)
        nm (apply m/new-matrix shape)]
    (replace-block m (-> nm (m/fill val) m/to-nested-vectors))))

(defn normalize [v]
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

