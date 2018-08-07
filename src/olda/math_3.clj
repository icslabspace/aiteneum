(ns olda.math-3
  (:require [fastmath.core :refer [digamma exp]]
            [clojure.walk :refer [postwalk]]
            [clojure.core.matrix :refer [esum emap add sub transpose vec? array?] :as m]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]
            [uncomplicate.neanderthal.core :as uncle]
            [uncomplicate.neanderthal.native :as untive]
            [uncomplicate.fluokitten.core :as fluc]
            [uncomplicate.fluokitten.jvm :as fluj]))


(defn expn-cm
  "Compute e^(ai) where ai are inputs of a 2D matrix;
  it returns a 2D matrix"
  [array]
  (emap #(exp %) array))

(defn ^:ndeanderthal di-gamma ^double [^double x]
  (digamma x))

(defn ^:neanderthal p+ ^double [^double x ^double y]
  (+ x y))

(defn ^:neanderthal p* ^double [^double x ^double y]
  (* x y))

(defn ^:neanderthal pdiv ^double [^double x ^double y]
  (/ x y))

(defn ^:neanderthal psi [m]
  (fluc/fmap di-gamma m))

(defn ^:neanderthal psi-of-sum [m]
  (fluc/fmap fluc/fold (uncle/rows m)))

(defn ^:slow-neanderthal expectation [a]
  (map (fn [r] (let [psi-sum (-> r uncle/sum digamma)]
                 (map #(-> %
                           digamma
                           (- psi-sum))
                      r)))
       (uncle/rows a)))

(defn ^:neanderthal pexpn ^double [^double x]
  (exp x))

(defn ^:neanderthal expn [a]
  (fluc/fmap pexpn a))

;; matrix operations
(defn mf [f mx] (fluc/fmap (fn ^double [^double x] (f x)) mx))

(defn mf! [f mx] (fluc/fmap! (fn ^double [^double x] (f x)) mx))

(defn mop [f val mx] (fluc/fmap (fn ^double [^double x] (f x val)) mx))

(defn mop! [f val mx] (fluc/fmap! (fn ^double [^double x] (f x val)) mx))

(defn m+ [val m] (mop + val m))

(defn m* [val m] (mop * val m))

(defn mdiv [val m] (mop / val m))

(defn m+! [val m] (mop! + val m))

(defn m*! [val m] (mop! * val m))

(defn mdiv! [val m] (mop! / val m))

(defn mmop [f mx my]
  (fluc/fmap (fn ^double [^double x ^double y] (f x y)) mx my))

(defn mmop! [f mx my]
  (fluc/fmap! (fn ^double [^double x ^double y] (f x y)) mx my))

(defn mm+ [mx my] (mmop + mx my))

(defn mm* [mx my] (mmop * mx my))

(defn mm+! [mx my] (mmop! + mx my))

(defn mm*! [mx my] (mmop! * mx my))

(defn outer-p [v1 v2] (pmap (fn [x] (map #(* x %) v2)) v1))

(defn outer-pt [v1 v2] (pmap (fn [x] (map #(* x %) v1)) v2))

(defn normalize
  "Given a collection of number values do a basic normalization"
  [v]
  (let [sv (apply + v)]
    (into [] (map #(/ % sv)) v)))

