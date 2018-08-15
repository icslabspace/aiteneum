(ns eva.bag-of-words
  (:require [clojure.core.reducers :as r]
            [clojure.string :as s]
            [clojure.java.io :as io]
            [eva.vocabulary :as vocab]))

(defn- update-bow [t-bow word]
  (let [idx (or (t-bow (keyword word)) 0)]
    (assoc! t-bow (keyword word) (inc idx))))

(defn- bow-to-lists [bow]
  {:word-ids (->> bow
                  keys
                  (into [])
                  (r/map name)
                  (r/map read-string)
                  (into []))
   :word-counts (->> bow
                     vals
                     (into []))})

(defn file->doc
  "Read the content of a text file and outputs
  a vector of strings"
  [file]
  (let [alpha-fn #(s/replace % #"[^\s\w]" "")
        tok-fn #(s/split % #"\s+")
        words-fn (fn [x] (filter #(not (= "" %)) x))]
    (-> file
        slurp
        alpha-fn
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
  {:word-ids [10 20 30 .. ]
   :word-counts [1 2 3 ..]}

  The collection at key :word-ids says what vocabulary
  tokens are present the input collection.

  The collection at key :word-counts says how many times
  the token appears int the input collection."
  
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
  indexed bag of words as list of maps <word-ids,word-counts>:
  [{:word-ids [10 20 30] :word-counts [1 2 3]} ...] 

  Map i from result, gives information about word ids and
  word counts from doc i from input"
  
  [docs]
  (loop [result (transient [])
         doc (first docs)
         d docs]
    (if (nil? doc)
      (persistent! result)
      (recur (conj! result (doc->indexed-bow doc))
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
    (if-not (nil? files)
      (->> files
           (pmap file->doc)
           (docs->indexed-bows))
      (throw (java.io.FileNotFoundException. (str folder " not found!"))))))

(defn ibow->doc
  "vocab-fn is the closure on vocabulary. The lambda fn has 1 arg
  which is the index of a element and returns the element at the given
  index."
  [ibow vocab-fn]
  (loop [result (transient [])
         words (:word-ids ibow)
         counts (:word-counts ibow)]
    (if (nil? (first words))
      (->> result persistent! flatten vec (map vocab-fn))
      (recur (conj! result (repeat (first counts) (first words)))
             (rest words)
             (rest counts)))))

(defn doc->file
  "Writes the doc (collection of strings) in the
  file with the given filename. Creates parent directories
  if needed in the path."
  ([doc filename]
   (clojure.java.io/make-parents filename)
   (spit filename "" :append false) 
   (doall (pmap #(spit filename (str % " ") :append true) doc))))

(defn indexed-bow->file
  "Converts an indexed bag of words into a collection of
  strings which is later written in a file. For the index to
  word convertion, the vocab-fn is needed.
  i.e.
  * if vocab is a list => vocab-fn is (partial nth vocab)
  * if vocab is a map => vocab-fn is (partial contains? vocab)
  * if vocab is eva.vocab => vocab-fn is (partial eva.vocabulary/get-word)

  If no folder is provided, a new folder is generated in the
  current path. Otherwise, the given folder is used as path
  for the new file."
  ([ibow vocab-fn]
   (ibow->file ibow
               vocab-fn
               (str "folder-" (.toString (java.util.UUID/randomUUID)))))
  ([ibow vocab-fn folder]
   (-> ibow
       (ibow->doc vocab-fn)
       (doc->file (str folder
                       java.io.File/separator
                       "file-"
                       (.toString (java.util.UUID/randomUUID)))))))
