(ns olda.dirichlet
  (:require [olda.math :as olda-math]
            [clojure.core.matrix :refer [esum emap sub transpose vec?]]))

(defn expectation
  "For an array theta ~ Dir(alpha) computes E[log(theta)] given alpha"
  [array]
  (let [psi-a (olda-math/psi array)
        psi-sum (olda-math/psi-of-sum array)]
    (if (vec? array)
      (sub psi-a psi-sum)
      (vec (map-indexed #(emap - %2 (nth psi-sum %1)) psi-a)))))


(defn xlogexp
  "Compute the expectation over an input array and then do an exponentiation
  the array's elements"
  [array]
  (-> array expectation olda-math/expn))
