(ns olda.core
  (:require [incanter.stats :refer [sample-gamma]]
            [clojure.core.reducers :as r]
            [olda.em :as em]
            [olda.math :as math]
            [olda.utils :as otils]))

(defn- complement [params docs agnostic-dict]  
  (let [params (assoc-in params [:model :estimated-num-docs] (count docs))
        params (assoc-in params [:model :dict :num-words] (-> agnostic-dict last inc))]
    params))

(defn- build-agnostic-dict [docs]
  (reduce #(into %1 (:word-ids %2)) (sorted-set) docs))

(defn train [params docs]
  (let [params (complement params docs (build-agnostic-dict docs))
        n (-> params :ctrl :m-iters)
        lambda (-> params :model em/sample-lambda')]
    (em/do-ems params docs lambda n)))

(defn update [model doc & docs])

(defn reap-topics [model doc-x]
  (-> model :gamma (nth doc-x) math/normalize))

(defn take-words [model topic-x top-n]
  (->> model :lambda (otils/nth' topic-x)
       (sort (comp - compare)) math/normalize
       (take top-n)))

(defn describe [model doc-index])
