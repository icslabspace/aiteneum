(ns olda.corpus
  (:require 
   [clojure.java.io :refer [resource] :as io]))

(defn extract-path
  "Given a string file path-to-file/file-name extract path-to-file/"
  [s]
  
  (re-find #".*/" s))

(defn extract-name
  "Givena strinng path-to-file/file-name extract file-name"
  [s]
  
  (re-find #"[^/]*$" s))

(defn get-path
  "Given a vector of documents' representations and document index get its path"
  [idx corpora]
  (:path (nth corpora idx)))

(defn get-name
  "Given a vector documents' representations and document index get its name"
  [idx corpora]
  (:name (nth corpora idx)))

(defn get-idx
  "Given a vector of document's representations, a document's path and name  get its index in the vector"
  [path name corpora]
  (loop [corpora corpora
         index 0]
    (if (and (-> corpora first :path (= path))
             (-> corpora first :path (= name)))
      index
      (recur (rest corpora)
             (inc index)))))

(defn get-indices
  "Get all available document indices for a Mallet model's data"
  [corpora]
  (-> corpora count range))

(defn read-file [f]
  (slurp f))

;; load and init a corpora based on dicretcory path

(defn struct-corpora
  "Given a directory path return a vector of document representations
  i.e. [{:path directory-path :name a-document's-file-name :pproc-fns :empty}
        ...]"
  
  [dir-path]
  (let [files (.listFiles (io/file dir-path))]
    (into []
          (map #(hash-map :path (-> % str extract-path)
                          :name (-> % str extract-name)
                          :pproc-fns :empty))
          files)))

(defn make-corpora
  "See struct-corpora; this function also populets the :pproc-fns key either with the optional argument either with a default simple document-to-words reading function"
  [dir-path & pproc-fns]
  (let [pproc-fns (or pproc-fns [read-file])]
    (into []
          (map #(assoc % :pproc-fns (vec pproc-fns)))
          (struct-corpora dir-path))))

(defn pre-process
  "Given a file path and a collection of preprocessing funcions preprocess the document and returns a vector of words"
  [file-path pproc-fns]
  ((apply comp (reverse pproc-fns)) file-path))

(defn read-and-pproc
  "Given a corpus representation get a vector of words"
  [doc]
  (let [file-path (str (:path doc) (:name doc))
        pproc-fns (:pproc-fns doc)
        result (pre-process file-path pproc-fns)]
    result))


