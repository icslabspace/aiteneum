(ns recommender.models.cofi
  {:author "Alex Gherega" :doc "Collaborative filtering recommender model"}
  
  (:require [clojure.core.matrix :as m]
            [fastmath.core :as fmc]
            [fastmath.stats :as fms]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]))

(m/set-current-implementation :vectorz)

(comment "in this model think of the x y r and theta matrices as follows:
E - entities of type for which one wishes a recommendation (e.g. movies); we refer to an entitye of this type as e 
U - entities that gave a ranking/score for elements in E (e.g. users); we refer to an entity of this type as u

x <- [(count E) x number of features] - a row in x is a feature vector for e
y <- [(count E) X (count U)] - a row in y is a scoring given by all u in U to an e
r <- same shape as y; whenever y(i,j) is 0 r(i,j) is 0.0; else r(i,j) is 1.0 
theta <- [(count U) x number of features] - a row in theta is features' weights vector for u

e.g.
x=[movies X features] where features could be genres; features are learned to depict a movie
y=[movies X users] to each movie a user gave a score [1..5] or 0 if no score
r=syntactic sugar for y with 1s and 0s
theta=[users X features] for each user a set of weights are learned to describe user preferences;
x * thetaT gives our predictions (a row will state users' preferences for a movie); or one could use x as a vector space an lookup similar movies by some distance metric")


;; cost function
(defn linear-cost [x y r theta]
  (let [sparse (m/mul (m/sub (m/mmul x (m/transpose theta)) y)
                      r)]
    
   
    (/ (m/esum (m/mul sparse sparse))
       2)))

(defn regularize-cost [x theta lambda init-cost]
  ;; original
  ;; (+ init-cost (* (/ lambda 2) (+ (m/esum (m/diagonal (m/mmul theta (m/transpose theta))))
  ;;                                 (m/esum (m/diagonal (m/mmul x (m/transpose x)))))))

  ;; efficient?
  (+ init-cost
     (* lambda 0.5 (+ (apply + (pmap m/dot theta theta))
                      (apply + (pmap m/dot x x))))))

(defn regularized-linear-cost [x y r theta lambda]
  (regularize-cost x theta lambda (linear-cost x y r theta)))

;; grandient functions

;; gradient computation for the features space
(defmacro gradient-xline [theta x-row y-row r-row]
  `(let [;;original
         ;; tmp-theta (m/mul theta (m/transpose (m/broadcast r-row (-> theta m/shape reverse))))
         ;; efficient?
         tmp-theta# (mapv m/scale ~theta ~r-row)
         ]
     (m/mmul (m/sub (m/mmul ~x-row (m/transpose tmp-theta#))
                    ~y-row)
             tmp-theta#)))

(defn- scale-byzero [m binary-v]
  (m/sparse-matrix (mapv #(if (zero? %2) (m/scale %1 0)
                              %1)
                         m
                         binary-v)))

(defn- gradient-xline [theta x-row y-row r-row]
  (let [;;original
        ;; tmp-theta (m/mul theta (m/transpose (m/broadcast r-row (-> theta m/shape reverse))))
        ;; efficient?
        tmp-theta (scale-byzero theta r-row) ;;(mapv m/scale theta r-row)
        ]
    (m/mmul (m/sub (m/mmul x-row (m/transpose tmp-theta))
                   y-row)
            tmp-theta)))

(defn grad-x [x y r theta]
  (pmap (partial gradient-xline theta) x y r))


;; gradient computation for the hyperparams space
(defn- gradient-thetaline [x y-col r-col theta-row]
  (let [;; original
        ;; tmp-x (m/mul x (m/transpose (m/broadcast r-col (-> x m/shape reverse))))
        ;;efficient?
         tmp-x (scale-byzero x r-col) ;;(mapv m/scale x r-col)
        ;; _ (prn 
        ;;        ;; (first (m/mmul tmp-x theta-row))
        ;;    ;; (filter #(.equals % ##NaN) (m/sub (m/mmul tmp-x theta-row)
        ;;    ;;                                   y-col))
        ;;    ;; (first (m/mmul (m/sub (m/mmul tmp-x theta-row)
        ;;    ;;                       y-col)
        ;;    ;;                tmp-x))
        ;;    (filter #(.equals % ##NaN) y-col)
        ;;    )
        ]
    (m/mmul (m/sub (m/mmul tmp-x theta-row)
                   y-col)
            tmp-x)))

(defn grad-theta [x y r theta]
  (pmap (partial gradient-thetaline x) 
        y ;; y is actualy in transpose form
        r ;; r is actualy in transpose form
        theta))

(defn regularize-grad [varm lambda init-grad]
  (m/add init-grad (m/scale varm lambda)))

;; regularized graiend functions
(defn regularized-gradx [x y r theta lambda]    
  (regularize-grad x lambda (grad-x x y r theta)))

(defn regularized-gradtheta [x y r theta lambda]
  (regularize-grad theta lambda (grad-theta x
                                            (m/transpose y) ;;y 
                                            (m/transpose r) ;;r
                                            theta)))

;; training by gradient descent
(defn- update-rule [in in-gradient alpha]
  (m/sub in (m/scale in-gradient alpha)))

(defn gd-train [x y r theta
                {:keys [rcost-f rgx-f rgtheta-f] :as r-fns}
                {:keys [lambda alpha epsilon no-iters] :as params}
                ;;rcost-f rgx-f rgtheta-f n
                ] ;; naive gradient descent trainig

  (loop [;; results
         x x
         theta theta
         n (:no-iters params)
         prev-cost 0]
    (prn "Iteration no: " n
         (-> theta first first)
         (-> x first first))
    (if (or (zero? n)
            (< (fmc/abs (- prev-cost (rcost-f x y r theta lambda)))
               epsilon))
      (do (prn "Iterations spent: " n) [x theta])
      (recur (update-rule x (rgx-f x y r theta lambda) alpha)
             (update-rule theta (rgtheta-f x y r theta lambda) alpha)
             (dec n)
             (rcost-f x y r theta lambda)))))


;; -------------------------------------
;; specs
(s/def ::binary-vector (s/coll-of #{0.0 1.0 0 1} :kind m/vec?))

(s/def ::binary-matrix (s/coll-of ::binary-vector :kind m/matrix?)
  )
(s/def ::same-shape-yr #(m/same-shape? (:y %) (:r %)))

(s/def ::shape-of-x #(= (-> % :x m/shape) [(-> % :y m/shape first)
                                           (-> % :theta m/shape second)]))

(s/def ::shape-of-theta #(= (-> % :theta m/shape) [(-> % :y m/shape second)
                                                   (-> % :x m/shape second)]))

(s/def ::x-theta #(= (-> % :x m/shape second)
                     (-> % :theta m/shape second)))
(s/fdef linear-cost
  :args (s/and (s/cat :x m/matrix?
                      :y m/matrix?
                      :r ::binary-matrix
                      :theta m/matrix?)
               
               ::same-shape-yr
               ::shape-of-x
               ::shape-of-theta)
  :ret number?)

(s/fdef regularize-cost
  :args (s/and (s/cat :x m/matrix?
                      :theta m/matrix?
                      :lambda number?
                      :init-cost number?)
               ::x-theta)
  :ret number?)

(s/fdef regularize-linear-cost ;; higher function composing previous two
  :args (s/and (s/cat :x m/matrix?
                      :y m/matrix?
                      :r ::binary-matrix
                      :theta m/matrix?
                      :lambda number?))
  :ret number?)


(s/fdef scale-byzero
  :args (s/and (s/cat :m m/matrix?
                      :binary-v ::binary-vector)
               #(= (-> % :m m/shape first)
                   (-> % :binary-v m/shape first)))
  :ret (s/and m/sparse? m/matrix?)
  :fn #(m/same-shape? (-> % :args :m)
                      (:ret %)))


(s/fdef gradient-xline
  :args (s/and (s/cat :theta m/matrix?
                      :x-row m/vec?
                      :y-row m/vec?
                      :r-row ::binary-vector)
               ))

;; (defn- gradient-xline [theta x-row y-row r-row]
;;   (let [;;original
;;         ;; tmp-theta (m/mul theta (m/transpose (m/broadcast r-row (-> theta m/shape reverse))))
;;         ;; efficient?
;;         tmp-theta (scale-byzero theta r-row) ;;(mapv m/scale theta r-row)
;;         ]
;;     (m/mmul (m/sub (m/mmul x-row (m/transpose tmp-theta))
;;                    y-row)
;;             tmp-theta)))

;; (defn grad-x [x y r theta]
;;   (pmap (partial gradient-xline theta) x y r))


;; ;; gradient computation for the hyperparams space
;; (defn- gradient-thetaline [x y-col r-col theta-row]
;;   (let [;; original
;;         ;; tmp-x (m/mul x (m/transpose (m/broadcast r-col (-> x m/shape reverse))))
;;         ;;efficient?
;;          tmp-x (scale-byzero x r-col) ;;(mapv m/scale x r-col)
;;         ;; _ (prn 
;;         ;;        ;; (first (m/mmul tmp-x theta-row))
;;         ;;    ;; (filter #(.equals % ##NaN) (m/sub (m/mmul tmp-x theta-row)
;;         ;;    ;;                                   y-col))
;;         ;;    ;; (first (m/mmul (m/sub (m/mmul tmp-x theta-row)
;;         ;;    ;;                       y-col)
;;         ;;    ;;                tmp-x))
;;         ;;    (filter #(.equals % ##NaN) y-col)
;;         ;;    )
;;         ]
;;     (m/mmul (m/sub (m/mmul tmp-x theta-row)
;;                    y-col)
;;             tmp-x)))

;; (defn grad-theta [x y r theta]
;;   (pmap (partial gradient-thetaline x) 
;;         y ;; y is actualy in transpose form
;;         r ;; r is actualy in transpose form
;;         theta))

;; (defn regularize-grad [varm lambda init-grad]
;;   (m/add init-grad (m/scale varm lambda)))

;; ;; regularized graiend functions
;; (defn regularized-gradx [x y r theta lambda]    
;;   (regularize-grad x lambda (grad-x x y r theta)))

;; (defn regularized-gradtheta [x y r theta lambda]
;;   (regularize-grad theta lambda (grad-theta x
;;                                             (m/transpose y) ;;y 
;;                                             (m/transpose r) ;;r
;;                                             theta)))

;; ;; training by gradient descent
;; (defn- update-rule [in in-gradient alpha]
;;   (m/sub in (m/scale in-gradient alpha)))

;; (defn gd-train [x y r theta
;;                 {:keys [rcost-f rgx-f rgtheta-f] :as r-fns}
;;                 {:keys [lambda alpha epsilon no-iters] :as params}
;;                 ;;rcost-f rgx-f rgtheta-f n
;;                 ] ;; naive gradient descent trainig

;;   (loop [;; results
;;          x x
;;          theta theta
;;          n (:no-iters params)
;;          prev-cost 0]
;;     (prn "Iteration no: " n
;;          (-> theta first first)
;;          (-> x first first))
;;     (if (or (zero? n)
;;             (< (fmc/abs (- prev-cost (rcost-f x y r theta lambda)))
;;                epsilon))
;;       (do (prn "Iterations spent: " n) [x theta])
;;       (recur (update-rule x (rgx-f x y r theta lambda) alpha)
;;              (update-rule theta (rgtheta-f x y r theta lambda) alpha)
;;              (dec n)
;;              (rcost-f x y r theta lambda)))))


(stest/instrument [;;`linear-cost `regularize-cost `regularized-linear-cost
                   ;;`scale-byzero
                   ;;`gradient-xline `gradient-thetaline
                   ;;`grad-x `grad-theta
                   ;;`regularize-grad
                   ;;`update-rule
                   ])
