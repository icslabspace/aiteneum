(ns olda.em-2
  (:require [clojure.core.reducers :as r]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mops]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]
            [incanter.stats :refer [sample-gamma]]
            [fastmath.stats :as fast-math]
            [fastmath.core :as fast-core]
            [olda.math :as olda-math]
            [olda.dirichlet :as dirichlet]
            [uncomplicate.neanderthal.core :as uncle]
            [uncomplicate.neanderthal.native :as untive]
            [uncomplicate.neanderthal.vect-math :as unmath]
            [uncomplicate.fluokitten.core :as fluc]
            [uncomplicate.fluokitten.jvm :as fluj]
            ))


(def model {:params {:ctrl {:counter 0
                            :num-iters 400
                            :epsilon 1e-3
                            :mean-thresh 1e-3}
                     :model {:alpha 0.5
                             :eta 2
                             :tau 3
                             :kappa 4
                             :num-topics 10
                             :estimated-num-docs 10
                             :dict {:num-words 4}
                             :gamma {:shape 100
                                     :scale 1e-2}}}
            :lambda nil
            :gamma nil})

(defn sample-gamma' [params docs]
  (let [r (count docs)
        c (:num-topics params)]
    (untive/dge r c (sample-gamma (* r c)
                                  :shape (-> params :gamma :shape)
                                  :scale (-> params :gamma :scale))
                {:layout :row})))

(defn sample-lambda' [params]
  (let [r (:num-topics params)
        c (-> params :dict :num-words)]
    (untive/dge r c (sample-gamma (* r c)
                                  :shape (-> params :gamma :shape)
                                  :scale (-> params :gamma :scale))
                {:layout :row})))

(defn- init-teta [params]
  (untive/dge (-> params :num-topics)
              (-> params :dict :num-words)
              {:layout :row}))

(defn gime-cols [m ids]
  ;; get all columns from m based on ids and return a neanderthal Matrix
  (time (untive/dge (map #(into [] (uncle/col m %)) ids)))) ;; use into []

(defn list->matrix [l]
  (let [[r c] (m/shape l)]
    (untive/dge r c l {:layout :row})))

(defn list->vec [l]
  (untive/dv l))

(defn norm-phi [eps xlog-thetad xlog-betad]
;;  (prn eps xlog-thetad xlog-betad)
  (map #(fluc/fmap (fn ^double [^double x] (+ eps x))
                   (uncle/dot xlog-thetad %))
       (uncle/rows xlog-betad)))

(defn- inside-gammad [params cts gammad xlog-thetad xlog-betad]
  (uncle/copy!
   (olda-math/m+ (-> params :model :alpha)
                 (olda-math/mm* ;(untive/dv xlog-thetad)
                                xlog-thetad
                                (untive/dv (map #(uncle/dot cts %)
                                                (uncle/cols xlog-betad)))))
   gammad))

(defn- inside-step [params ids cts gammad xlog-thetad xlog-betad stats]

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

  (loop [results (repeatedly 100
                             (fn [] (let [ ;; keep last gamma for mean threshold change verification
                                          last-gammad (uncle/copy gammad)
                                          
                                          ;;update gammad
                                          _ (inside-gammad params cts gammad xlog-thetad xlog-betad)
                                          _ (uncle/copy! (-> gammad dirichlet/xlogexp list->vec) xlog-thetad)
                                          ;;(uncle/axpby! 1.0 (-> gammad dirichlet/xlogexp list->vec) 0.0 xlog-thetad)
                                          ]
                                      
                                      (/ (uncle/asum (uncle/axpy -1.0 last-gammad gammad))
                                         (uncle/dim last-gammad)))))]
    
    (if (or (-> results seq not)
            (< (first results) 1.0e-35))
      (doall (map #(do (olda-math/mm+! (uncle/col stats %1) %2))
                  ids
                  (uncle/cols (untive/dge (uncle/dim xlog-thetad)
                                          (uncle/dim cts)
                                          (olda-math/outer-p xlog-thetad (unmath/div cts
                                                                                     (untive/dv (norm-phi (-> params :ctrl :epsilon) xlog-thetad xlog-betad))))
                                          {:layout :row}))))
      ;; update xlog-thetad
      (recur (do ;(prn (first results))
               (rest results)
               )))))


;; todo: moveo this function to somewhere else
(defn docs->neanderthal [docs]
  (map #(zipmap (keys %)
                (map untive/dv (vals %)))
       docs))

(defn do-e [params docs lambda stats]
  (let [gamma (sample-gamma' (:model params) docs)
        xlog-theta (list->matrix (dirichlet/xlogexp gamma))
        xlog-beta (list->matrix (dirichlet/xlogexp lambda))
        ;; this next bit: mutable stuff!
        _ (doall
           (map (fn [doc gammad xlog-thetad]
                  (inside-step params
                               (:word-ids doc)
                               (:word-counts doc)
                               gammad                        
                               xlog-thetad
                               (gime-cols xlog-beta (:word-ids doc))
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
        ;; lro (olda-math/m*! (->> params :rho (- 1.0)) lambda)
        ;; expectation (time (olda-math/m* (:estimated-num-docs params)
        ;;                                 (olda-math/mdiv docs-count stats)))
        ;; expectation (time (olda-math/m*! (:rho params)
        ;;                                  (olda-math/m+! (:eta params) expectation)))
        ;; expectation (time (olda-math/mf
        ;;                    (fn [x] (* rho (+ eta (* D (/ x docs-count)))))
        ;;                    stats))
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
         stats (untive/dge D W (repeat  (* D W) 0) {:layout :row})
         
         ;; update params
         params (assoc params :model model-params)
         
         ;; do e-step
         ;_ (prn "DBUG: starting E-step")
         gs (do-e params docs lambda stats)

         ;; do m-step
         ;_ (prn "DBUG: starting M-step")         
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
