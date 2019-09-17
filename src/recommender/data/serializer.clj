(ns recommender.data.serializer
  {:author "Alex Gherega" :doc "Basic serialization operations"}
  
  (:require [clojure.core.matrix :as m]
            [clojure.java.io :as io]
            [clojure.edn :as edn]))

(defn m->clj [m]
  (m/to-nested-vectors m))

(defn baby-spit [m x-fn filepath]
  (with-open [w (io/writer filepath)]
    (.write w (-> m x-fn pr-str))))

(defn baby-slurp [filepath]
  (with-open [r (io/reader filepath)]
    (edn/read (java.io.PushbackReader. r))))
