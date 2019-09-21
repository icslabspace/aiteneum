(ns recommender.data.serializer
  {:author "Alex Gherega" :doc "Basic serialization operations"}
  
  (:require [clojure.core.matrix :as m]
            [clojure.java.io :as io]
            [clojure.edn :as edn]
            [clojure.spec.alpha :as s]))

(defn m->clj [m]
  (m/to-nested-vectors m))

(defn baby-spit [m x-fn filepath]
  (with-open [w (io/writer filepath)]
    (.write w (-> m x-fn pr-str))))

(defn baby-slurp [filepath]
  (with-open [r (io/reader filepath)]
    (edn/read (java.io.PushbackReader. r))))


;; ---------------------------------------
;; spec
(s/def ::ymean (s/or :col-mat m/column-matrix?
                     :number number?))

(s/def ::model  (s/and vector?
                       (s/cat :x m/matrix? :theta m/matrix? :ymean ::ymean)
                       #(m/same-shape? (-> % :x first)
                                       (-> % :theta first))))
(s/fdef m->clj
  :args (s/cat :m m/matrix?)
  :ret (s/coll-of vector? :kind vector?)
  :fn (s/and #(= (-> % :args :m m/shape)
                 [(-> % :ret count) (-> :ret first count)])
             #(every? (fn [v] (= (count v)
                                 (-> % :args :m m/shape second)))
                      (:ret %))))

(s/fdef baby-spit
  :args (s/cat :m ::model :x-fn ifn? :filepath string?))

(s/fdef baby-slurp
  :args (s/cat :filepath string?)
  :ret ::model)
