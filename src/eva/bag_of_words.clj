(ns eva.bag-of-words
  (:require [clojure.core.reducers :as r]
            [clojure.string :as st]
            [clojure.java.io :as io]
            [eva.vocabulary :as vocab]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]))

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
  (let [alpha-fn #(st/replace % #"[^\s\w]" "")
        tok-fn #(st/split % #"\s+")
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

(defn indexed-bow->doc
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
   (indexed-bow->file ibow
                      vocab-fn
                      (str "folder-" (.toString (java.util.UUID/randomUUID)))))
  ([ibow vocab-fn folder]
   (-> ibow
       (indexed-bow->doc vocab-fn)
       (doc->file (str folder
                       java.io.File/separator
                       "file-"
                       (.toString (java.util.UUID/randomUUID)))))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; SPECS ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; regex
(def folder-name-regex #"([.]{2}[/]{1})*[[\w[\\-]{0,1}]+/]+")
(def file-name-regex #"([.]{2}[/]{1})*[[\w[\\-]{0,1}]+/]+[\w[\\-]{0,1}[(]*[)]*]+[[.]{1}txt]{0,1}")

(s/def ::filename (s/and string?
                         #(re-matches file-name-regex %)))
(s/def ::folder (s/and string?
                       #(re-matches folder-name-regex %)))

(s/def ::doc (s/coll-of string?))
(s/def ::docs (s/coll-of ::doc))

(s/def ::word-ids (s/coll-of int?))
(s/def ::word-counts (s/coll-of int?))
(s/def ::bow (s/coll-of (s/tuple string? int?)))
(s/def ::ibow (s/keys :req-un [::word-ids ::word-counts]))
(s/def ::ibows (s/coll-of ::ibow))
(s/def ::vocab-fn ifn?) ;; (s/fspec :args (s/cat :idx int?) :ret string?)

;; functions working on collection of strings 
(s/fdef doc->bow
  :args (s/cat :doc ::doc)
  :ret ::bow)

(s/fdef doc->indexed-bow
  :args (s/cat :doc ::doc)
  :ret ::ibow)

(s/fdef docs->indexed-bows
  :args (s/cat :docs ::docs)
  :ret ::ibows)

(s/fdef indexed-bow->doc 
  :args (s/cat :doc ::ibow
               :vocab-fn ::vocab-fn)
  :ret ::doc)

;; functions working on files
(s/fdef file->doc
  :args (s/alt :file ::filename
               :file #(instance? java.io.File %)) 
  :ret ::doc)

(s/fdef file->indexed-bow
  :args (s/cat :file ::filename)
  :ret ::ibow)

(s/fdef files->indexed-bows
  :args (s/cat :folder ::folder)
  :ret ::ibows)

(s/fdef doc->file
  :args (s/cat :doc ::doc
               :filename ::filename)
  :ret coll?)

(s/fdef indexed-bow->file
  :args (s/cat :ibow ::ibow
               :vocab-fn ::vocab-fn
               :folder (s/* ::folder))
  :ret coll?)

;; instrumentation
(stest/instrument [`doc->bow
                   `doc->indexed-bow
                   `docs->indexed-bows
                   `indexed-bow->doc
                   `file->doc
                   `file->indexed-bow
                   `files->indexed-bow
                   `doc->file
                   `indexed-bow->files])



