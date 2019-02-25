(ns aiteneum.core
  (:require [olda.core :as olda]
            [eva.bag-of-words :as eva]))

(def docs (read-string (slurp "resources/docs.txt")))

(def i-bow (eva.bag-of-words/docs->indexed-bows docs))

(def params {:ctrl {:counter 0
                    :m-iters 1
                    :num-iters 200
                    :epsilon 1e-3
                    :mean-thresh 1e-5}
             :model {:alpha 0.5
                     :eta 2
                     :tau 3
                     :kappa 4
                     :num-topics 8
                     :estimated-num-docs nil
                     :dict {:num-words nil}
                     :gamma {:shape 100
                             :scale 1e-2}}})

;(def olda-model (olda/train params i-bow))

;(def i-bow (eva.bag-of-words/files->indexed-bows "../--all/-/resources/data/single-doc/"))
