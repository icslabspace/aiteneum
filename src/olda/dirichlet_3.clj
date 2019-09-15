(ns olda.dirichlet-3
  (:require [olda.math :as olda-math]
            [fastmath.core :as fmath]
            [clojure.core.matrix :refer [esum emap sub transpose vec?] :as m]
            [uncomplicate.neanderthal.core :as uncle]
            [uncomplicate.neanderthal.native :as untive]
            [uncomplicate.fluokitten.core :as fluc]
            [uncomplicate.fluokitten.jvm :as fluj]))

(defn ^:neanderthal vxlogexp
  "For a v ~ Dir(*) compute e^E[log(v)] where
  v is an OpenCL vector"
  [v] ;; v is a OpenCL vector
  (let [psi-sum (-> v uncle/sum fmath/digamma)]
    (fluc/fmap! (fn ^double [^double x]
                  (-> x
                      fmath/digamma
                      (- psi-sum)
                      fmath/exp))
                v)))

(defn ^:neanderthal axlogexp
  "For an a ~ Dir(*) compute e^E[log(a)] where
  a is an OpenCL matrix"
  [a]
  (doall (pmap vxlogexp (uncle/rows a)))
  a)

(defn ^:neanderthal xlogexp
  "Compute e^E[log(a)] for an OpenCL array (1D or 2D)
  where a represents a Dir(*) distribution"
  [a]
  (cond (uncle/vctr? a) (vxlogexp a)
        (uncle/matrix? a) (axlogexp a)))

