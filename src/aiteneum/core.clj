(ns aiteneum.core
  (:require [olda.core :as olda]
            [eva.bag-of-words :as eva]))

(def docs (read-string (slurp "resources/docs.txt")))

(def i-bow (eva.bag-of-words/docs->indexed-bows docs))

(def params {:ctrl {:counter 0.0
                    :m-iters 100
                    :num-iters 100
                    :epsilon 1e-100
                    :mean-thresh 1e-35}
             :model {:alpha (/ 1.0 8.0)
                     :eta (/ 1.0 8.0)
                     :tau 1.0
                     :kappa 0.5
                     :num-topics 8
                     :estimated-num-docs 8
                     :dict {:num-words 8228}
                     :gamma {:shape 100
                             :scale 1e-2}}})

;(def olda-model (olda/train params i-bow))

;(def i-bow (eva.bag-of-words/files->indexed-bows "../cambioscience-all/selena/resources/data/single-doc/"))

;(def i-bow (eva.bag-of-words/files->indexed-bows "resources/"))

