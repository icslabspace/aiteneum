(ns eva.bag-of-words
  (:require [clojure.core.reducers :as r]
            [clojure.string :as s]
            [clojure.java.io :as io]
            [eva.vocabulary :as vocab]))

(defn- update-bow [t-bow word]
  (let [idx (or (t-bow (keyword word)) 0)]
    (assoc! t-bow (keyword word) (inc idx))))

(defn- bow-to-lists [bow]
  {:ids (->> bow
             keys
             (into [])
             (r/map name)
             (r/map read-string)
             (r/fold conj))
   :counts (->> bow
                vals
                (into []))})

(defn- merge-stats [x y]
  (-> x
      (update :idss conj! (:ids y))
      (update :countss conj! (:counts y))))

(defn file->doc
  "Read the content of a text file and outputs
  a vector of strings"
  [file]
  (let [tok-fn #(s/split % #"\s+")
        words-fn (fn [x] (filter #(not (= "" %)) x))]
    (-> file
        slurp
        tok-fn
        words-fn)))

(defn doc->bow
  "Convert a collection of strings to a bag of words
  (ie. an association between a word (as keyword) and the
  number of appearences in the input)"
  [doc]
  (loop [bow (transient {})
         words doc]
    (if (-> words first nil?)
      (persistent! bow)
      (recur (update-bow bow (-> words first))
             (rest words)))))

(defn doc->indexed-bow
  "Convert a collection of strings into an indexed bag
  of words as map of lists:
  {:ids `(10 20 30 .. )
   :counts `(1 2 3 ..)}

  The list at key :ids says what vocabulary tokens are present
  the input collection.

  The list at key :counts says how many times the token appears
  int the input collection."
  
  [doc]
  (loop [bow (transient {})
         word (first doc)
         words doc]
    (if (nil? word)
      (bow-to-lists (persistent! bow))
      (do
        (if-not (vocab/has-word? word)
          (vocab/add-word word))
        (recur (update-bow bow (vocab/get-idx-str word))
               (second words)
               (rest words))))))

(defn docs->indexed-bows
  "Convert a collection of collection of strings into an
  indexed bag of words as a map of lists of lists:
  {:idss `((10 20 30) (30 40 10) .. )
   :countss `((1 2 3) (3 4 1) ..)}

  The list at key :idss says what vocabulary tokens are
  present in each document. idss[i][j] gives the jth
  unique token present in document i. 
  
  The list at key :countss says how many times each vocabulary
  token is present. counts[i][j] is the number of times
  that the token given by wordids[i][j] appears in document i."
  
  [docs]
  (loop [result {:idss (transient [])
                 :countss (transient [])}
         doc (first docs)
         d docs]
    (if (nil? doc)
      {:idss (persistent! (:idss result))
       :countss (persistent! (:countss result))}
      (recur (merge-stats result (doc->indexed-bow doc))
             (second d)
             (rest d)))))

(defn file->indexed-bow
  "Read the file with the given filename and
  convert the content to an index bag of words"
  [filename]
  (-> filename
      file->doc
      doc->indexed-bow))

(defn files->indexed-bows
  "Read the files from the given folder and
  convert the content into indexed bags of words"
  [folder]
  (let [files (-> folder
                  io/file
                  .listFiles)]
    (->> files
         (pmap file->doc)
         (docs->indexed-bows))))

;; includes duplicates
(defn get-bags-size [i-bows]
  (reduce #(+ %1 (reduce + 0 %2)) 0 (:counts i-bows)))


