(ns recommender.models.cofi
  {:author "Alex Gherega" :doc "Collaborative filtering recommender model"}
  
  (:require [clojure.core.matrix :as m]
            [fastmath.core :as fmc]
            [fastmath.stats :as fms]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]
            [recommender.particles :as parts]))

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
(defn linear-cost
  "x - MxK matrix;
  y - MxN matrix;
  r - MxN matrix;
  theta - NxK matrix"
  [x y r theta]
  (let [sparse (m/mul (m/sub (m/mmul x (m/transpose theta)) y)
                      r)]
    
   
    (/ (m/esum (m/mul sparse sparse))
       2)))

(defn regularize-cost
  "x - MxK matrix;
  theta - NxK matrix;
  lambda and init-cost are just numbers"
  [x theta lambda init-cost]
  ;; efficient
  (+ init-cost
     (* lambda 0.5 (+ (apply + (pmap m/dot theta theta))
                      (apply + (pmap m/dot x x))))))

(defn regularized-linear-cost
  "Basic regularized linear cost function for Collaborative Filtering learning
  x - MxK matrix;
  y - MxN matrix;
  r - MxN matrix;
  theta - NxK matrix"
  [x y r theta lambda]
  (regularize-cost x theta lambda (linear-cost x y r theta)))

;; grandient functions

;; gradient computation for the features space
(defmacro gradient-xline
  [theta x-row y-row r-row]
  `(let [;;original
         ;; tmp-theta (m/mul theta (m/transpose (m/broadcast r-row (-> theta m/shape reverse))))
         ;; efficient?
         tmp-theta# (mapv m/scale ~theta ~r-row)
         ]
     (m/mmul (m/sub (m/mmul ~x-row (m/transpose tmp-theta#))
                    ~y-row)
             tmp-theta#)))

(defn- scale-byzero
  [m binary-v]
  (m/sparse-matrix (mapv #(if (zero? %2) (m/scale %1 0)
                              %1)
                         m
                         binary-v)))

(defn- gradient-xline
  "x-row a row from x - a MxK matrix,
  y-row a row from y - a MxN matrix;
  r-row a row from r - a MxN matrix;
  theta - NxK matrix"

  [theta x-row y-row r-row]
  (let [;;original
        ;; tmp-theta (m/mul theta (m/transpose (m/broadcast r-row (-> theta m/shape reverse))))
        ;; efficient?
        tmp-theta (scale-byzero theta r-row) ;;(mapv m/scale theta r-row)
        ]
    (m/mmul (m/sub (m/mmul x-row (m/transpose tmp-theta))
                   y-row)
            tmp-theta)))

(defn grad-x
  "x - MxK matrix;
  y - MxN matrix;
  r - MxN matrix;
  theta - NxK matrix"
  [x y r theta]
  (pmap (partial gradient-xline theta) x y r))


;; gradient computation for the hyperparams space
(defn- gradient-thetaline
  "x - a MxK matrix,
  y-col a column from y - a MxN matrix;
  r-col a column from r - a MxN matrix;
  theta-row a row from theta - a NxK matrix"
  [x y-col r-col theta-row]
  (let [tmp-x (scale-byzero x r-col) ;;(mapv m/scale x r-col)
        ]
    (m/mmul (m/sub (m/mmul tmp-x theta-row)
                   y-col)
            tmp-x)))

(defn grad-theta
  "x - MxK matrix;
  y - MxN matrix;
  r - MxN matrix;
  theta - NxK matrix"
  [x y r theta]
  (pmap (partial gradient-thetaline x) 
        y ;; y is actualy in transpose form such that pmap traverses columns of initial y
        r ;; r is actualy in transpose form
        theta))

(defn regularize-grad
  "varm a matrix to be scaled by lambda;
  add the resulting scaled matrix elemnt-wise to the gradient matrix init-grad"
  [varm lambda init-grad]
  (m/add init-grad (m/scale varm lambda)))

;; regularized graiend functions
(defn regularized-gradx
  "x - MxK matrix;
  y - MxN matrix;
  r - MxN matrix;
  theta - NxK matrix"
  [x y r theta lambda]    
  (regularize-grad x lambda (grad-x x y r theta)))

(defn regularized-gradtheta
  "x - MxK matrix;
  y - MxN matrix;
  r - MxN matrix;
  theta - NxK matrix"
  [x y r theta lambda]
  (regularize-grad theta lambda (grad-theta x
                                            (m/transpose y) ;;y 
                                            (m/transpose r) ;;r
                                            theta)))

;; training by gradient descent
(defn- update-rule
  [in in-gradient alpha]
  (m/sub in (m/scale in-gradient alpha)))

(defn gd-train
  "Your run of th emill basic and trivial gradient descent;
  x - MxK matrix;
  y - MxN matrix;
  r - MxN matrix;
  theta - NxK matrix;
  rcost-f a cost function, rgx-f gradient of the cost function w.r.t x function, rgtheta-f gradient of the cost w.r.t theta
  params - required to tune the training algorithm"
  [x y r theta
   {:keys [rcost-f rgx-f rgtheta-f] :as r-fns}
   {:keys [lambda alpha epsilon no-iters] :as params}]

  ;; naive gradient descent trainig

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
(s/def ::binary-iter :recommender.particles/binary-iter ;;#(every? #{1.0 0.0 1 0} %)
  )

(s/def ::binary-vector :recommender.particles/binary-vector
  ;; (s/and m/vec?
  ;;        #(every? #{1.0 0.0 1 0} %))
  )

(s/def ::binary-matrix :recommender.particles/binary-matrix
  ;; (s/and m/matrix?
  ;;        #(every? (fn [row] (s/valid? ::binary-vector row)) %))
  )

(s/def ::same-shape-yr #(m/same-shape? (:y %) (:r %)))

(s/def ::shape-of-x #(= (-> % :x m/shape) [(-> % :y m/shape first)
                                           (-> % :theta m/shape second)]))

(s/def ::shape-of-theta #(= (-> % :theta m/shape) [(-> % :y m/shape second)
                                                   (-> % :x m/shape second)]))

(s/def ::x-theta #(= (-> % :x m/shape second)
                     (-> % :theta m/shape second)))

(s/def ::rcost-f #{recommender.models.cofi/regularized-linear-cost})
(s/def ::rgx-f #{recommender.models.cofi/regularized-gradx})
(s/def ::rgtheta-f #{recommender.models.cofi/regularized-gradtheta})

(s/def ::lambda number?)
(s/def ::alpha number?)
(s/def ::epsilon pos?)
(s/def ::no-iters pos-int?)

(s/def ::params (s/keys :req-un [::lambda ::alpha ::epsilon ::no-iters]))

(s/def ::r-fns (s/keys :req-un [::rcost-f ::rgx-f ::rgtheta-f]))

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
                      :lambda ::lambda
                      :init-cost number?)
               ::x-theta)
  :ret number?)

(s/fdef regularize-linear-cost ;; higher function composing previous two
  :args (s/and (s/cat :x m/matrix?
                      :y m/matrix?
                      :r ::binary-matrix
                      :theta m/matrix?
                      :lambda ::lambda))
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
               #(let [[n k] (-> % :theta m/shape)]
                  (and (= (-> % :x-row m/shape) [k])
                       (= (-> % :y-row m/shape) (-> % :r-row m/shape) [n]))))
  :ret m/vec?
  :fn #(m/same-shape? (-> % :args :x-row)
                      (-> % :ret)))

(s/fdef grad-x
  :args (s/and (s/cat :x m/matrix?
                      :y m/matrix?
                      :r ::binary-matrix
                      :theta m/matrix?)
               
               ::same-shape-yr
               ::shape-of-x
               ::shape-of-theta)
  :ret (s/coll-of m/vec?)
  :fn #(m/same-shape? (-> % :args :x)
                      (-> % :ret)))

(s/fdef gradient-thetaline
    :args (s/and (s/cat :x m/matrix?
                        :y-col m/vec?
                        :r-col ::binary-vector
                        :theta-row m/vec?)
                 #(let [[m k] (-> % :x m/shape)]
                    (and (= (-> % :theta-row m/shape) [k])
                         (= (-> % :y-col m/shape) (-> % :r-col m/shape) [m]))))
    :ret m/vec?
    :fn #(m/same-shape? (-> % :args :theta-row)
                        (-> % :ret)))

(s/fdef grad-theta
  :args (s/and (s/cat :x m/matrix?
                      :yt m/matrix?  ;; this is a transposed matrix;
                      :rt ::binary-matrix  ;; this is a tranposed matrix
                      :theta m/matrix?)
               
               ::same-shape-yr               
               #(= (-> % :x m/shape) [(-> % :yt m/shape second)
                                      (-> % :theta m/shape second)])
               #(= (-> % :theta m/shape) [(-> % :yt m/shape first)
                                          (-> % :x m/shape second)]))
  
  :ret (s/coll-of m/vec?)
  :fn #(m/same-shape? (-> % :args :theta)
                      (-> % :ret)))

(s/fdef regularize-grad
  :args (s/and (s/cat :varm m/matrix?
                      :lambda ::lambda
                      :init-grad m/matrix?)
               #(m/same-shape? (:varm %) (:init-grad %)))
  :ret m/matrix?
  :fn #(m/same-shape? (-> % :args :init-grad)
                      (-> % :ret)))

(s/fdef update-rule
  :args (s/and (s/cat :in m/matrix
                      :in-gradient m/matrix?
                      :alpha ::alpha)
               #(m/same-shape? (:in %) (:in-gradient %)))
  :ret m/matrix?
  :fn #(m/same-shape? (-> % :args :in)
                      (:ret %)))

(s/fdef gd-train
  :args (s/and (s/cat :x m/matrix?
                      :y m/matrix?
                      :r ::binary-matrix
                      :theta m/matrix?
                      :fns ::r-fns
                      :params ::params)
               ::same-shape-yr
               ::shape-of-x
               ::shape-of-theta)
  :ret (s/coll-of m/matrix? :kind vector?)
  :fn #(and (m/same-shape? (-> % :args :x) (-> % :ret first))
            (m/same-shape? (-> % :args :theta) (-> % :ret second))))

;; ;; spec instrumentation is off as it impacts performance time
;; (stest/instrument [`linear-cost `regularize-cost `regularized-linear-cost
;;                    `scale-byzero
;;                    `gradient-xline `gradient-thetaline
;;                    `grad-x `grad-theta
;;                    `regularize-grad
;;                    `update-rule
;;                    `gd-train
;;                    ])
