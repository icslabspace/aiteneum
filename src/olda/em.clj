(ns olda.em
  (:require [clojure.core.reducers :as r]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :as mops]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]
            [incanter.stats :refer [sample-gamma]]
            [fastmath.stats :as fast-math]
            [fastmath.core :as fast-core]
            [olda.math :as olda-math]
            [olda.dirichlet :as dirichlet]))

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

(defn- norm-phi [eps xlog-tetad xlog-betad]
  (mops/+ (m/dot xlog-tetad xlog-betad)
          eps))

(def a (atom nil))
(def wc (atom nil))
(def xt (atom nil))
(def xb (atom nil))
(def p (atom nil))

(defn update-gammad [alpha word-cts xlog-tetad xlog-betad phinorm]
  (if (nil? @a)
    (do (reset! a alpha)
        (reset! wc word-cts)
        (reset! xt xlog-tetad)
        (reset! xb xlog-betad)
        (reset! p phinorm)))
  ;; (let [word-cts (:word-cts doc)
  ;;       alpha (:alpha params)])
  (mops/+ alpha
          (mops/* xlog-tetad
                  (m/dot (mops// word-cts phinorm)
                         (m/transpose xlog-betad)))))

(defn ^:make-private gammad-teta-phi
  
  [params doc xlog-tetad xlog-betad phinorm]
  (let [gammad (update-gammad (-> params :model :alpha)
                              (:word-counts doc)
                              xlog-tetad xlog-betad phinorm)
        xlog-tetad (dirichlet/xlogexp gammad)
        phinorm (norm-phi (-> params :ctrl :epsilon) xlog-tetad xlog-betad)]
    
    [gammad xlog-tetad phinorm]))

(defn- mean-changed? [prev-gammad curr-gammad mean-thresh]
  (-> (mops/- curr-gammad prev-gammad)
      m/abs
      fast-math/mean
      (< mean-thresh)))

(defn- make-betad [word-ids xlog-beta]
  (m/transpose (into []
                     (map #(m/get-column xlog-beta %))
                     word-ids)))

(defn- converge-gamma-phi
  ([params doc gammad xlog-betad]
   ;; elogd ->  {:xlog-theatad exp-elog-tetad :xlog-betad exp-elog-betad}
   
   (loop [n (-> params :ctrl :num-iters)
          prev-gammad gammad
          [curr-gammad xlog-tetad phinorm] (gammad-teta-phi params doc (dirichlet/xlogexp gammad)
                                                            xlog-betad
                                                            (norm-phi (-> params :ctrl :epsilon)
                                                                      (dirichlet/xlogexp gammad)
                                                                      xlog-betad))]
     
     (if (or (mean-changed? prev-gammad curr-gammad (-> params :ctrl :mean-thresh))               
             (zero? n))
       (do (prn "DBUG: " n) [curr-gammad (m/outer-product xlog-tetad (mops// (:word-counts doc) phinorm))])
       (recur (dec n)
              curr-gammad
              (gammad-teta-phi params doc xlog-tetad xlog-betad phinorm)))))
  
  ([idx params docs gamma xlog-beta]
   (converge-gamma-phi params
                       (first docs)
                       (m/get-row gamma idx)
                       (make-betad (-> docs first :word-ids) xlog-beta))))


(defn- update-teta-old [ids columns teta]
  (let [res
        (reduce #(let [idx (nth ids %2)
                       new-col (mops/+ (m/get-column %1 idx)
                                       (m/get-column columns %2))]
                   (m/set-column %1 idx new-col))
                teta (-> ids count range))]
    res))


(defn- reducer-fn [ids columns shape]
  (fn
    ([] (apply m/new-matrix shape))
    ([zero-m id] (m/set-column zero-m
                               (nth ids id)
                               (m/get-column columns id)))))

(defn- combiner-fn [shape]
  (fn
    ([] (apply m/new-matrix shape))
    ([m1 m2] (mops/+ m1 m2))))

(def i (atom nil))
(def c (atom nil))
(def t (atom nil))

(defn update-teta [ids columns teta]
  (if (nil? @i)
    (do (reset! i ids) (reset! c columns) (reset! t teta)))
  (let [shape (m/shape teta)
        reduce-f (reducer-fn ids columns shape)
        combine-f (combiner-fn shape)
        sparse-m (r/fold combine-f reduce-f (-> ids count range))]
    (mops/+ teta sparse-m)))


(defn sample-gamma' [params docs]
  (repeatedly (count docs)
              (partial sample-gamma
                       (-> params :num-topics)
                       :shape (-> params :gamma :shape)
                       :scale (-> params :gamma :scale))))

(defn sample-lambda' [params]
  (repeatedly (-> params :num-topics)
              (partial sample-gamma
                       (-> params :dict :num-words)
                       :shape (-> params :gamma :shape)
                       :scale (-> params :gamma :scale))))

(defn- init-teta [params]
  (m/new-matrix (-> params :num-topics)
                (-> params :dict :num-words)))

(defn- compute-rho [tau kappa counter]
  (fast-core/pow (+ tau counter)
                 (- kappa)))
(defn do-e
  "The e step of the Online LDA algorithm"
  ([params docs gamma lambda]
   (let [;gamma (-> params :model (sample-gamma' docs))
         teta (-> params :model init-teta)
         xlog-beta (dirichlet/xlogexp lambda)
         stats (converge-gamma-phi 0 params docs gamma xlog-beta)]
     
     (do-e params docs gamma teta xlog-beta stats)))

  ([params docs lambda]
   (let [gamma (-> params :model (sample-gamma' docs))
         teta (-> params :model init-teta)
         xlog-beta (dirichlet/xlogexp lambda)
         stats (converge-gamma-phi 0 params docs gamma xlog-beta)]
     
     (do-e params docs gamma teta xlog-beta stats)))
  
  ([params docs gamma teta xlog-beta stats]
   (loop [idx 0 ;;batch-size (count docs)
          doc (first docs)
          docs (rest docs)
          gamma gamma
          xlog-beta xlog-beta
          teta teta
          stats stats]
     (if (-> docs seq not)
       [;gamma (mops/* teta xlog-beta)
        (m/set-row gamma idx (first stats))
        (mops/* (update-teta (:word-ids doc) (second stats) teta) xlog-beta)]
       (recur (inc idx) (first docs) (rest docs)
              (m/set-row gamma idx (first stats))
              xlog-beta
              (do (prn "DBUG-start-update: " (inc idx)) (update-teta (:word-ids doc) (second stats) teta))
              (do (prn "DBUG-start-converge: " (inc idx)) (converge-gamma-phi (inc idx) params docs gamma xlog-beta)))))))


(defn do-m
  "The m step of the Online LDA algorithm"
  [params docs-count lambda [gamma stats]]
  (let [lro (mops/* (->> params :rho (- 1)) lambda)
        nstats (mops/* (:estimated-num-docs params)
                       (mops// stats docs-count))]
    (mops/+ lro (mops/* (:rho params)
                        (mops/+ (:eta params) nstats)))))

(defn do-em
  "The Online LDA training algorithm"
  ([params docs lambda]
   (let [;; init some model params
         model-params (assoc (:model params)
                             :rho (compute-rho (-> params :model :tau)
                                               (-> params :model :kappa)
                                               (-> params :ctrl :counter))
                             :tau (-> params :model :tau inc))
         ;; update params
         params (assoc params :model model-params)
         
         ;; do e-step
         _ (prn "DBUG: starting E-step")
         [gamma stats :as gs] (do-e params docs lambda)
         
         ;; do m-step
         _ (prn "DBUG: starting M-step")         
         lambda (do-m (:model params) (count docs) lambda gs)

         _ (prn "DBUG: all done!")] 
     
     ;; return updated latent vars
     {:params (update-in params [:ctrl :counter] inc)
      :gamma gamma
      :lambda lambda}))
  
  ([params docs gamma lambda]
   (let [;; init some model params
         model-params (assoc (:model params)
                             :rho (compute-rho (-> params :model :tau)
                                               (-> params :model :kappa)
                                               (-> params :ctrl :counter))
                             :tau (-> params :model :tau inc))
         ;; update params
         params (assoc params :model model-params)
         
         ;; do e-step
         [gamma stats :as gs] (do-e params docs gamma lambda)
         
         ;; do m-step
         lambda (do-m (:model params) (count docs) lambda gs)] 
     
     ;; return updated latent vars
     {:params (update-in params [:ctrl :counter] inc)
      :gamma gamma
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

(defn grow-model [model doc-cts num-words num-topics] ;; doc-cts, word-cts and num-topics need to be larger than model's
  (let [model (assoc-in model
                        [:params :model :estimated-num-docs]
                        (-> model :params :model :estimated-num-docs (+ doc-cts)))
        model (assoc-in model [:params :model :dict :num-words] num-words)
        nu-lambda (sample-lambda' (-> model :params :model))]
    
    (assoc model :lambda (olda.math/replace-block (:lambda model) nu-lambda))))

;; specs

(s/def ::word string?)
(s/def ::pos-int (s/or ::pos pos? ::zero zero?))
(s/def ::word-batch (s/map-of ::word ::pos-int))

(s/def ::xlog-tetad m/array?) ;; usually this is a vector
(s/def ::xlog-betad m/array?)
(s/def elogd (s/keys :req-un [::xlog-tetad ::xlog-betad]))

(s/fdef norm-phi
        :args (s/cat :xlog-tetad ::xlog-tetad
                     :xlog-betad ::xlog-betad
                     :epsilon double?)
        :ret double?)
(s/fdef do-e
        :args (s/cat :doc (s/coll-of ::word-batch))
        :ret any?)
