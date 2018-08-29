(ns olda.em-3
  (:require [clojure.core.reducers :as r]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mops]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]
            [incanter.stats :refer [sample-gamma]]
            [fastmath.stats :as fast-math]
            [fastmath.core :as fast-core]
            [olda.math-3 :as olda-math]
            [olda.dirichlet-3 :as dirichlet]
            [uncomplicate.neanderthal.core :as uncle]
            [uncomplicate.neanderthal.native :as untive]
            [uncomplicate.neanderthal.vect-math :as unmath]
            [uncomplicate.neanderthal.opencl :refer [with-default-engine] :as unopen]
            [uncomplicate.fluokitten.core :as fluc]
            [uncomplicate.fluokitten.jvm :as fluj]
            [midje.sweet :refer [facts => truthy]]
            [criterium.core :refer [quick-bench with-progress-reporting]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl.core :refer [with-default finish!]]
            ))

(def model {:params {:ctrl {:counter 0.0
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
                                     :scale 1e-2}}}
            :lambda nil
            :gamma nil})

(defmacro sample-gammal [dim shape scale]
  `(if (> ~dim 1)
     (sample-gamma ~dim
                   :shape ~shape
                   :scale ~scale)
     [(sample-gamma ~dim
                    :shape ~shape
                    :scale ~scale)]))

(defn sample-gamma' [params docs]
  (let [r (count docs)
        c (:num-topics params)]
    (untive/dge r c (sample-gammal (* r c)
                                   (-> params :gamma :shape)
                                   (-> params :gamma :scale))
                {:layout :row})))

(defn sample-lambda' [params]
  (let [r (:num-topics params)
        c (-> params :dict :num-words)]
    (untive/dge r c (sample-gammal (* r c)
                                   (-> params :gamma :shape)
                                   (-> params :gamma :scale))
                {:layout :row})))

(defn- init-teta [params]
  (untive/dge (-> params :num-topics)
              (-> params :dict :num-words)
              {:layout :row}))

(defn ^:obsolete gime-cols [m ids] ;; this is expensive
  ;; get all columns from m based on ids and return a neanderthal Matrix
  (untive/dge (pmap #(into [] (uncle/col m %)) ids)))

(defn gime-v [m f ids] ;; f can only be uncle/row or uncle/col
  (map #(f m %) ids))

(defn gime-rows [m ids] ;; <======== CONSUMING
  ;; get all columns from m based on ids and return a neanderthal Matrix
  (apply pmap untive/dv (gime-v m uncle/col ids))
  ;;(gime-v m uncle/col ids)
  ) ;; use into []

(defn gime-cols [m ids]
  ;; get all columns from m based on ids and return a neanderthal Matrix
  (gime-v m uncle/col ids))

(defn list->matrix [l]
  (let [[r c] (m/shape l)]
    (untive/dge r c l {:layout :row})))

(defn list->vec [l]
  (untive/dv l))

(defn norm-phi [eps xlog-thetad xlog-betad]
  ;;  compute a normalizing value for Phi
  (pmap #(+ eps (uncle/dot xlog-thetad %))
        xlog-betad))

(defn- inside-gammad! [params cts gammad xlog-thetad xlog-betadT]
  ;; ATTENTION: mutable area
  (uncle/copy!
   (fluc/fmap (fn ^double [^double x ^double y]
                (+ (-> params :model :alpha) (* x y)))
              xlog-thetad (untive/dv (pmap #(uncle/dot cts %)
                                           xlog-betadT)))
   gammad))

(defn update-gammad! [params cts gammad xlog-thetad xlog-betadT]
  ;; ATTENTION: mutable area
  (let [;; keep gamma for mean
        ;; threshold change verification
        last-gammad (uncle/copy gammad)]
    
    ;; update gammad
    (inside-gammad! params cts gammad xlog-thetad xlog-betadT)
    (uncle/copy! (-> gammad uncle/copy dirichlet/xlogexp) xlog-thetad)
    
    ;; compute mean difference
    (/ (uncle/asum (uncle/axpy -1.0 last-gammad gammad))
       (uncle/dim last-gammad))))

(defn- inside-step! [params ids cts gammad xlog-thetad xlog-betad xlog-betadT stats]

  (loop [results (repeatedly (-> params :ctrl :num-iters)
                             (partial update-gammad! params cts gammad xlog-thetad xlog-betadT))
         meanthresh (-> params :ctrl :mean-thresh)]
    
    (if (or (-> results seq not)
            (< (first results) meanthresh))
      ;; <=============== consuming
      (doall (let [phinorm (unmath/div cts    
                                       (untive/dv
                                        (norm-phi (-> params :ctrl :epsilon)
                                                  xlog-thetad xlog-betad)))]
               
               (map-indexed #(uncle/axpby! (uncle/entry phinorm %1)
                                           xlog-thetad
                                           1.0
                                           (uncle/col stats %2))
                            ids)))
      ;; update xlog-thetad
      (recur (rest results)
             (* 1e5 meanthresh)))))


    ;; todo: moveo this function to somewhere else


(defn docs->neanderthal [docs]
  (pmap #(zipmap (keys %)
                 (pmap untive/dv (vals %)))
        docs))

(defn do-e! [params docs lambda stats]
  (let [gamma (sample-gamma' (:model params) docs)
        xlog-theta (-> gamma uncle/copy dirichlet/xlogexp)
        xlog-beta (-> lambda uncle/copy dirichlet/xlogexp)]
    ;; this next bit: mutable stuff!
    (doall
     (map (fn [doc gammad xlog-thetad]
            (inside-step! params
                          (:word-ids doc)
                          (:word-counts doc)
                          gammad                        
                          xlog-thetad
                          (gime-cols xlog-beta (:word-ids doc))
                          (gime-rows xlog-beta (:word-ids doc)) ;; 8 seconds could be lost in this if we would not need to recreate nanderthal vector ds
                          stats))
          docs (uncle/rows gamma) (uncle/rows xlog-theta)))
    (olda-math/mm*! stats xlog-beta)
    
    ;; returning stats and gamma - although this could be avoided as both stats and gamma were muted in place
    {:stats stats
     :gamma gamma}))

(defn compute-rho [tau kappa counter]
  (fast-core/pow (+ tau counter)
                 (- kappa)))

(defn do-m!
  "The m step of the Online LDA algorithm"
  [params docs-count lambda gamma stats]
  (let [rho (:rho params)
        eta (:eta params)
        D (:estimated-num-docs params)
        b (- 1.0 rho)
        expectation (uncle/alter! (uncle/copy stats)
                                  (fn ^double [^long i ^long j ^double x]
                                    (* rho (+ eta (* D (/ x docs-count))))))]

    (uncle/axpby! 1.0 expectation b lambda)))

(defn do-em!
  "The Online LDA training algorithm"
  ([params docs lambda]
   (let [;; init some model params
         params (assoc-in params [:model :rho]
                          (compute-rho (-> params :model :tau)
                                       (-> params :model :kappa)
                                       (-> params :ctrl :counter)))
         
         K (-> params :model :num-topics)
         W (-> params :model :dict :num-words)
         stats (untive/dge K W (repeat  (* K W) 0) {:layout :row})
         
         ;; do e-step
         gs (do-e! params docs lambda stats)

         ;; do m-step
         ;; ATTENTION: mutable stuff
         _ (do-m! (:model params) (count docs) lambda
                  (:gamma gs)
                  (:stats gs))] 
     
     ;; return updated latent vars
     {:params (update-in params [:ctrl :counter] inc)
      :gamma (:gamma gs)
      :stats (:stats gs)
      :lambda lambda})))

(defn do-ems!
  "This is what usually is called to train an
  Online LDA model for a number of n iterations;

  ATTENTION: lambda will be mutated by the end"

  [params docs lambda n]
  (loop [iters n
         model {:params params
                :gamma nil
                :lambda lambda}]
    (if (zero? iters)
      model
      (recur (dec iters)
             (do-em! (:params model) docs (:lambda model))))))
    
    
