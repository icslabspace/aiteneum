(ns olda.utils
  (:require [clojure.core.matrix :as m]
            [uncomplicate.neanderthal
             [native :refer [dv dge fge dtr native-float]]
             [cuda :as cuda]
             [opencl :as ocl]
             [core :refer [copy copy! submatrix scal! transfer! transfer mrows ncols nrm2 mm cols view-tr] :as uncle]
             [real :refer [entry entry!]]
             [linalg :refer [trf tri det]]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl.core :as clojurecl]
            ))

(defn nth'
  "The right order of params for nth"
  [index coll]
  (nth coll index))

(defn get-row'
  "The right order of params for nth"
  [index coll]
  (m/get-row coll index))

(defn ^:neanderthal row
  "The right order of params for nth"
  [index coll]
  (uncle/row coll index))
