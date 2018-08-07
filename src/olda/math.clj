(ns olda.math
  (:require [fastmath.core :refer [digamma exp]]
            [clojure.walk :refer [postwalk]]
            [clojure.core.matrix :refer [esum emap add sub transpose vec? array?] :as m]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]
            [uncomplicate.neanderthal.core :as uncle]
            [uncomplicate.neanderthal.native :as untive]
            [uncomplicate.fluokitten.core :as fluc]
            [uncomplicate.fluokitten.jvm :as fluj]))

(defn ^:deprecated psi-d
  "Compute the psi function for a matrix input"
  [matrix]
  (postwalk #(if (number? %)
               (digamma %)
               %)
            matrix))

(defn ^:core-matrix psi-cm [matrix]
  (emap #(digamma %) matrix))

(defn ^:deprecated psi-of-sum-1
  "Add columns of a matrix and then compute
  the psi function on that vector"
  [matrix]
  (->> matrix
       ;; transpose and add is equivalent to: (into [] (map #(apply + %)) alpha)
       transpose
       (apply add)
       m/array
       psi-cm))

(defn psi-of-sum-m-cm
  [matrix]
  (->> matrix
       m/rows
       (map m/esum)
       m/array
       psi-cm))

(defn psi-of-sum-v-cm
  [v]
  (m/esum v))

(defn psi-of-sum-cm
  [a]
  (cond (m/vec? a) (psi-of-sum-v-cm a)
        (m/matrix? a) (psi-of-sum-m-cm a)))

(defn expn-cm
  "Compute e^(ai) where ai are inputs of a 2D matrix;
  it returns a 2D matrix"
  [array]
  (emap #(exp %) array))

(defn- ^:ndeanderthal di-gamma ^double [^double x]
  (digamma x))

(defn- ^:neanderthal P+ ^double [^double x ^double y]
  (+ x y))

(defn ^:neanderthal psi [m]
  (fluc/fmap di-gamma m))

(defn ^:neanderthal psi-of-sum [m]
  (fluc/fmap fluc/fold (uncle/rows m)))

(defn ^:neanderthal expectation [a]
  ;; (cond
  ;;   (uncle/vctr? a) (uncle/sum a)
  ;;   (uncle/matrix? a) (->> a uncle/rows (map uncle/sum)
  ;;                          untive/dv
  ;;                          psi))
  (map (fn [r] (let [psi-sum (-> r uncle/sum digamma)]
                 (map #(-> %
                           digamma
                           (- psi-sum))
                      r)))
       (uncle/rows a)))

(defn- ^:neanderthal exponential ^double [^double x]
  (exp x))

(defn ^:neanderthal expn [a]
  (fluc/fmap exponential a))

;; new stuff
(defn mf [f mx]
  (fluc/fmap (fn ^double [^double x] (f x)) mx))

(defn mf! [f mx]
  (fluc/fmap! (fn ^double [^double x] (f x)) mx))

(defn mop [f val mx]
  (fluc/fmap (fn ^double [^double x] (f x val)) mx))

(defn mop! [f val mx]
  (fluc/fmap! (fn ^double [^double x] (f x val)) mx))

(defn m+ [val m]
  (mop + val m))

(defn m* [val m]
  (mop * val m))

(defn mdiv [val m]
  (mop / val m))

(defn m+! [val m]
  (mop! + val m))

(defn m*! [val m]
  (mop! * val m))

(defn mdiv! [val m]
  (mop! / val m))

(defn mmop [f mx my]
  (fluc/fmap (fn ^double [^double x ^double y] (f x y)) mx my))

(defn mmop! [f mx my]
  (fluc/fmap! (fn ^double [^double x ^double y] (f x y)) mx my))

(defn mm+ [mx my]
  (mmop + mx my))

(defn mm* [mx my]
  (mmop * mx my))

(defn mm+! [mx my]
  (mmop! + mx my))

(defn mm*! [mx my]
  (mmop! * mx my))

(defn outer-p [v1 v2]
  (map (fn [x] (map #(* x %) v2)) v1))

(defn outer-pt [v1 v2]
  (map (fn [x] (map #(* x %) v1)) v2))

;; improve adataptability to the stochastic nature of the online training behaviour
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

