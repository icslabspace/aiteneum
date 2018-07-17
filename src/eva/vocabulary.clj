(ns eva.vocabulary)

;; TODO: in the future, replace this with
;; file/db storage & caching mechanism
(def vocab (atom {}))

(defn has-word? [word]
  (->> word
       keyword
       (contains? @vocab)))

(defn get-idx [word]
  (-> word
      keyword
      (@vocab)))

(defn get-idx-str [word]
  (-> word
      get-idx
      str))

(defn get-word [idx]
  (->> @vocab
       (filter #(= idx (val %)))
       first
       key
       name))

(defn add-word [word]
  (swap! vocab assoc (keyword word) (count @vocab)))

(defn size []
  (count @vocab))

