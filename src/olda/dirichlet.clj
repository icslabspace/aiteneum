(ns olda.dirichlet
  (:require [olda.math :as olda-math]
            [clojure.core.matrix :refer [esum emap sub transpose vec?]]))

(defn expectation
  "
  For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
  "
  [alpha]
  (let [psi-alpha (olda-math/psi alpha)
        psi-sum (olda-math/psi-of-sum alpha)]
    (if (vec? alpha)
      (sub psi-alpha psi-sum)
      (vec (map-indexed #(emap - %2 (nth psi-sum %1)) psi-alpha)))))


(defn xlogexp [alpha]
  (-> alpha expectation olda-math/expn))
