(ns olda.example
  (:require [aiteneum.core :as aiteneum]
            [clojure.core.matrix :as m]
            [olda.em-3 :as oldam]
            [eva.vocabulary :as evav]
            [olda.core :as oldac]))


(def params aiteneum/params)

(do (time (def res (oldam/do-ems
                    params
                    (oldam/docs->neanderthal aiteneum/i-bow)
                    (oldam/sample-lambda' (:model params))
                    (-> params :ctrl :m-iters))))
    (m/pm (map #(map evav/get-word
                     (oldac/take-words
                      res % 5))
               (range 8))))
