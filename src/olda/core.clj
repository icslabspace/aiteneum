(ns olda.core
  (:require [incanter.stats :refer [sample-gamma]]
            [clojure.core.reducers :as r]
            [clojure.core.matrix :as m]
            [olda.em-3 :as em]
            [olda.math-3 :as math]
            [olda.utils :as otils]))

(defn- complement
  "Update parameters with estimated number of documents
  and dictionary info (i.e. number of words)"
  [params docs agnostic-dict]
  (let [params (assoc-in params [:model :estimated-num-docs] (count docs))
        params (assoc-in params [:model :dict :num-words] (-> agnostic-dict last inc))]
    params))

(defn build-agnostic-dict
  "Given a corpus build an agnostic dictionary
  (i.e. knows only the id representation of words)"
  [docs]
  (reduce #(into %1 (:word-ids %2)) (sorted-set) docs))

(defn train
  "Given a set of model parameters and a corpus
  return a trained an Online LDA model"
  [params docs]
  (let [params (complement params docs (build-agnostic-dict docs))
        n (-> params :ctrl :m-iters)
        lambda (-> params :model em/sample-lambda')]
    (em/do-ems! params docs lambda n)))

(defn update
  "Given a trained Online LDA model
  update it using new corpus"
  [model doc & docs])

(defn reap-topics
  "Given an Online LDA model and a document index
  return topic distributions for that document"
  [model doc-x]
  (-> model :gamma (m/get-row doc-x) math/normalize))

(defn take-words
  "Given an Online LDA model and a topic index
  return the top-n words for that topic"
  [model topic-x top-n]
  (let [words-of-topic (->> model :lambda
                            ;(otils/nth' topic-x)
                            (otils/row topic-x)
                            (into []))
        sorted-vals (->> (into [] words-of-topic)
                         (sort (comp - compare))
                         ;math/normalize
                         (take top-n))]
    (map #(.indexOf words-of-topic %) sorted-vals)))

(defn describe
  "Give complete description of topic distributions
  and top-n words distributions given a certain Online LDA
  trained model and a document index"
  [model doc-x top-n])
