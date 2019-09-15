(ns olda.dirichlet
  (:require [olda.math :as olda-math]
            [fastmath.core :as fmath]
            [clojure.core.matrix :refer [esum emap sub transpose vec?] :as m]
            [uncomplicate.neanderthal.core :as uncle]
            [uncomplicate.neanderthal.native :as untive]
            [uncomplicate.fluokitten.core :as fluc]
            [uncomplicate.fluokitten.jvm :as fluj]))

(defn expectation-1
  "For an array theta ~ Dir(alpha) computes E[log(theta)] given alpha"
  [array]
  (let [psi-a (olda-math/psi array)
        psi-sum (olda-math/psi-of-sum array)]
    (if (vec? array)
      (sub psi-a psi-sum)
      (map-indexed #(emap - %2 (nth psi-sum %1)) psi-a))))


(defn xlogexp-1
  "Compute the expectation over an input array and then do an exponentiation
  the array's elements"
  [array]
  (-> array expectation-1 olda-math/expn m/array))

(defn ^:core-matrix expectation-cm
  "For an array theta ~ Dir(alpha) computes E[log(theta)] given alpha"
  [array]
  (let [psi-a (olda-math/psi array)
        psi-sum (olda-math/psi-of-sum array)]
    (if (vec? array)
      (sub psi-a psi-sum)
      (-> (emap sub
                                        ;(transpose psi-a)
                (-> psi-a m/columns m/array)
                psi-sum)
          transpose))))

(defn ^:core-matrix expectation-cm-1
  "For an array theta ~ Dir(alpha) computes E[log(theta)] given alpha"
  [array]
  (let [psi-a (olda-math/psi array)
        psi-sum (olda-math/psi-of-sum array)]
    (if (vec? array)
      (sub psi-a psi-sum)
      (-> (emap sub
                psi-a                      ;(-> psi-a m/columns m/array)
                (transpose psi-sum))
          transpose))))

(defn ^:core-matrix xlogexp-cm
  "Compute the expectation over an input array and then do an exponentiation
  the array's elements"
  [array]
  (-> array expectation-cm olda-math/expn-cm))


(defn ^:neanderthal expectation
  "For an array theta ~ Dir(alpha) computes E[log(theta)] given alpha"
  [a]
  (let [psi-a (olda-math/psi a)
        psi-sum (olda-math/psi-of-sum a)]
    (if (uncle/vctr? a)
      (uncle/axpy! -1 psi-sum psi-a)
      (-> (map sub
               (uncle/cols psi-a)                 ;(-> psi-a m/columns m/array)
               (transpose psi-sum))
          transpose))))

(defn ^:neanderthal expectation
  "For an array theta ~ Dir(alpha) computes E[log(theta)] given alpha"
  [a]
  (cond (uncle/vctr? a) (map #(-> %
                                  fmath/digamma
                                  (- (-> a uncle/sum fmath/digamma)))
                             a)
        (uncle/matrix? a) (map (fn [r] (let [psi-sum (-> r uncle/sum fmath/digamma)]
                                         (map #(-> %
                                                   fmath/digamma
                                                   (- psi-sum))
                                              r)))
                               (uncle/rows a))))

(defn ^:core-matrix xlogexp
  "Compute the expectation over an input array and then do an exponentiation
  the array's elements"
  [a]
  (-> a expectation-cm olda-math/expn))


(defn ^:neanderthal vxlogexp [v]
  (let [psi-sum (-> v uncle/sum fmath/digamma)]
    (map #(-> %
              fmath/digamma
              (- psi-sum)
              fmath/exp)
         v)))

(defn ^:neanderthal axlogexp [a]
  (map vxlogexp (uncle/rows a)))

(defn ^:neanderthal xlogexp
  "Compute the expectation over an input array and then do an exponentiation
  the array's elements"
  [a]
  (cond (uncle/vctr? a) (vxlogexp a)
        (uncle/matrix? a) (axlogexp a)))

