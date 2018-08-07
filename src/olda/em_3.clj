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
            [uncomplicate.clojurecl.core :refer [with-default finish!]]))

(with-default
  (with-default-engine))

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

(defn sample-gamma' [params docs]
  (let [r (count docs)
        c (:num-topics params)]
    (untive/dge
     ;;unopen/clge
     r c (sample-gamma (* r c)
                                   :shape (-> params :gamma :shape)
                                   :scale (-> params :gamma :scale))
     {:layout :row})))

(defn sample-lambda' [params]
  (let [r (:num-topics params)
        c (-> params :dict :num-words)]
    (untive/dge
     ;;unopen/clge
     r c (sample-gamma (* r c)
                       :shape (-> params :gamma :shape)
                       :scale (-> params :gamma :scale))
     {:layout :row})))

(defn- init-teta [params]
  (untive/dge
   ;;unopen/clge
   (-> params :num-topics)
   (-> params :dict :num-words)
   {:layout :row}))

(defn gime-cols [m ids]
  ;; get all columns from m based on ids and return a neanderthal Matrix
  (untive/dge
   ;;unopen/clge
   (pmap #(into [] (uncle/col m %)) ids))) ;; use into []

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
    (untive/dge
     ;;unopen/clge
     r c l {:layout :row}
     )))

(defn list->vec [l]
  (untive/dv l)
  ;;(unopen/clv l)
  )

(defn norm-phi [eps xlog-thetad xlog-betad]
  ;;  (prn eps xlog-thetad xlog-betad)
  (pmap #(+ eps (uncle/dot xlog-thetad %))
        xlog-betad))

(defn- inside-gammad [params cts gammad xlog-thetad xlog-betadT]
  (uncle/copy!
   (fluc/fmap (fn ^double [^double x ^double y]
                (+ (-> params :model :alpha) (* x y)))
              xlog-thetad (untive/dv
                           ;;unopen/clv
                           (pmap #(uncle/dot cts %)
                                 xlog-betadT)))
   ;; (olda-math/m+ (-> params :model :alpha)
   ;;               (olda-math/mm* xlog-thetad
   ;;                              (untive/dv
   ;;                               ;;unopen/clv
   ;;                               (pmap #(uncle/dot cts %)
   ;;                                     xlog-betadT))))
   gammad))

(defn- inside-step [params ids cts gammad xlog-thetad xlog-betad xlog-betadT stats]

  ;; ways to do it:
  ;; 1) mutual recursion between gammad-computation-fn and norm-phi: NO - recursion seems bad
  ;; 2) some sort of in-mem state of gammad, phinorm : BETTER
  ;;  (def m1 (unn/dge 2 3 [[1 1 1] [2 2 2]] {:layout :row}))
  ;;  (map #(unc/axpy! (unn/dv [1 2 3]) %) (unc/rows m1))
  ;; 2.1) we could use some neanderthal data structure which is mutable
  ;; 2.2) we could use clojure's references
  ;; recurence with loop/recur still seems the only true option
  ;; but it's soo expensive
  ;; 3) we shoud use a reduce call updating in memory gammad and phinorm; once meanthresh is reached just keep doing very inexpesive op

  (loop [results (repeatedly (-> params :ctrl :num-iters)
                             (fn [] (let [ ;; keep last gamma for mean threshold change verification
                                          last-gammad (uncle/copy gammad)
                                          
                                          ;;update gammad
                                          _ (inside-gammad params cts gammad xlog-thetad xlog-betadT)
                                          _ (uncle/copy! (-> gammad uncle/copy dirichlet/xlogexp) xlog-thetad)]
                                      
                                      (/ (uncle/asum (uncle/axpy -1.0 last-gammad gammad))
                                         (uncle/dim last-gammad)))))
         meanthresh (-> params :ctrl :mean-thresh)]
    
    (if (or (-> results seq not)
            (< (first results) meanthresh))
      ;; <=============== consuming
      (doall (let [phinorm (unmath/div cts    
                                          (untive/dv
                                           ;;unopen/clv
                                           (norm-phi (-> params :ctrl :epsilon) xlog-thetad xlog-betad)))]
                  (map-indexed #(uncle/axpby! (uncle/entry phinorm %1) xlog-thetad
                                              1.0
                                              (uncle/col stats %2))
                               ids)))
      ;; update xlog-thetad
      (recur (rest results)
             (* 1e5 meanthresh)))))


    ;; todo: moveo this function to somewhere else


(defn docs->neanderthal [docs]
  (pmap #(zipmap (keys %)
                 (pmap untive/dv
                       ;;unopen/clv
                       (vals %)))
        docs))

(defn do-e [params docs lambda stats]
  (let [gamma (sample-gamma' (:model params) docs)
        xlog-theta (-> gamma uncle/copy dirichlet/xlogexp)
        xlog-beta (-> lambda uncle/copy dirichlet/xlogexp)
        ;; this next bit: mutable stuff!
        _ (doall
           (map (fn [doc gammad xlog-thetad]
                  (inside-step params
                                      (:word-ids doc)
                                      (:word-counts doc)
                                      gammad                        
                                      xlog-thetad
                                      (gime-cols xlog-beta (:word-ids doc))
                                      (gime-rows xlog-beta (:word-ids doc));; 8 seconds lost in this 
                                      ;;(time (gime-cols xlog-beta (:word-ids doc)))
                                      stats))
                docs
                (uncle/rows gamma)
                (uncle/rows xlog-theta)))
        _ (olda-math/mm*! stats xlog-beta)]
    {:stats stats
     :gamma gamma}))

(defn compute-rho [tau kappa counter]
  (fast-core/pow (+ tau counter)
                 (- kappa)))

(defn do-m
  "The m step of the Online LDA algorithm"
  [params docs-count lambda gamma stats]
  (let [
        rho (:rho params)
        eta (:eta params)
        D (:estimated-num-docs params)
        b (- 1.0 rho)
        expectation (uncle/alter! (uncle/copy stats)
                                  (fn ^double [^long i ^long j ^double x]
                                    (* rho (+ eta (* D (/ x docs-count))))))]

    (uncle/axpby! 1.0 expectation b lambda)))

(defn do-em
  "The Online LDA training algorithm"
  ([params docs lambda]
   (let [;; init some model params
         model-params (assoc (:model params)
                             :rho (compute-rho (-> params :model :tau)
                                               (-> params :model :kappa)
                                               (-> params :ctrl :counter))
                                        ;:tau (-> params :model :tau inc)
                             )
         D (-> params :model :estimated-num-docs)
         W (-> params :model :dict :num-words)
         stats (untive/dge
                ;;unopen/clge
                D W (repeat  (* D W) 0) {:layout :row})
         
         ;; update params
         params (assoc params :model model-params)
         
         ;; do e-step
         ;;_ (prn "DBUG: starting E-step")
         gs (do-e params docs lambda stats)

         ;; do m-step
         ;;_ (prn "DBUG: starting M-step")         
         _ (do-m (:model params) (count docs) lambda
                 (:gamma gs)
                 (:stats gs))
         
         ;;_ (prn "DBUG: all done!")
         ] 
     
     ;; return updated latent vars
     {:params (update-in params [:ctrl :counter] inc)
      :gamma (:gamma gs)
      :stats (:stats gs)
      :lambda lambda})))

(defn do-ems
  "This is what usually is called to train an
  Online LDA model for a number of n iterations"
  [params docs lambda n]
  (loop [iters n
         model {:params params
                :gamma nil
                :lambda lambda}]
    (if (zero? iters)
      model
      (recur (dec iters)
             (do-em (:params model) docs (:lambda model))))))
    
    
