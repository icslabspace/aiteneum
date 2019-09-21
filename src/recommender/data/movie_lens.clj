(ns recommender.data.movie-lens
  {:author "Alex Gherega" :doc "From movie lens to Clojure data structures"}
  
  (:require [clojure.data.csv :as csv]
            [clojure.core.matrix :as m]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]
            [recommender.particles :as parts]))

(def num-features 19)

(def movies-csv "resources/recommender/movie-lens/ml-latest-small/movies.csv") ;; this has sturcture [movieId, title, genres]

(def ratings-csv "resources/recommender/movie-lens/ml-latest-small/ratings.csv") ;; this has structure [userId, movieId, rating, timestamp]

(def genres {:action "Action"
             :adventure "Adventure"
             :animation "Animation"
             :children "Children"
             :comedy "Comedy"
             :crime "Crime"
             :documentary "Documentary"
             :drama "Drama"
             :fantasy "Fantasy"
             :film-noir "Film-Noir"
             :horror "Horror"
             :musical "Musical"
             :mystery "Mystery"
             :romance "Romance"
             :sci-fi "Sci-Fi"
             :thriller "Thriller"
             :war "War"
             :western "Western"})

(defn safe-read-string
  "Read s into a Clojure data structure if possible; otherwise return the string"
  [s]
  (try (read-string s)
       (catch Exception e s)))

(defn read-number
  "Read the string and if it is a number return the number; otherwise return the original string"
  [s]
  (let [n? (safe-read-string s)]
    (if (number? n?) n? s)))

(defmulti read-lens-csv-row
  "Return a MovieLens csv row as a Clojure data structure;
  the structure of the returned data depends on the CSV file structure; hence the multifn"
  (fn [_ k] k))

(defmethod read-lens-csv-row :movies
  [[id title genre :as csv-row] _]
  (vector (read-number id) title genre))

(defmethod read-lens-csv-row :ratings
  [[uid movid rating timestamp :as csv-row] _]
  (mapv read-number csv-row))

(defn read-lens-csv
  "Traverse a MovieLens CSV file an call read-lens-csv-row on each line"
  [csv-file k]
  (pmap #(read-lens-csv-row % k)
        (drop 1 (with-open [reader (io/reader csv-file)]
                  (doall (csv/read-csv reader))))))

(defn count-entity
  "Given a collection of some sort of tuples, where the first element in each tuple is a number return the maximum of these numbers"
  [entities-full]
  (apply max (map first entities-full)))

;; ;; deprecated

;; (defn- ^:deprecated update-ymatrix [transient-result [user-id movie-id rating :as rating-line]]
;;   (assoc! transient-result
;;           (dec movie-id)
;;           (assoc (nth transient-result (dec movie-id)) (dec user-id) rating)))

;; (defn ^:deprecated  create-ymatrix [movies ratings]
;;   (let [num-movies (count movies)
;;         max-movie-id (count-entity movies)
;;         num-users (count-entity ratings)
;;         zero-yrow (zipmap (range num-users) (repeat num-users 0.0))
;;         ]
        
;;     ;;ymatrix
;;     (pmap #(->> % (merge zero-yrow) sort vals)
;;           (loop [ratings ratings
;;                  result (transient (into [] (repeat num-movies {})))]
;;             (if (-> ratings seq not)
;;               (persistent! result)
;;               (recur (rest ratings)
;;                      (update-ymatrix result (first ratings))))))
    
;;     ))

(defn get-movies
  "Filter a Clojure movies CSV representation by genre"
  [movies-csv genre]
  (filter #(->> % last (re-find (-> genres genre re-pattern)))
          (read-lens-csv movies-csv :movies)))

;; could try reducers
(defn get-ratings
  "Given a ratings csv data structure and a movies [csv] dataset return a collection of tuples of the following structure: [[userId movieId] rating]"
  [ratings-csv movies]
  (let [movies-ids (into #{} (map first movies))]
    (reduce #(if (some (-> %2 second hash-set) movies-ids)
               (assoc %1 [(first %2);; userId
                          (second %2);; movieId
                          ]
                      (nth %2 2) ;;movie rating
                      )
               %1)
            {}
            (read-lens-csv ratings-csv :ratings))))

(defn nativeids->matrixids
  "Given a set of native ids create a map with 0 - (count natives) keys and the original native ids as values"
  [natives natives-count selector-fn]
  (zipmap (sort (map selector-fn natives))
          (range natives-count)))

(defn matrixids->nativeids
  "See nativeids->matrixids; this returns a similar map but with keys as vals and vals as keys"
  [natives natives-count selector-fn]
  (zipmap (range natives-count)
          (sort (map selector-fn natives))))

(defn create-ymatrix
  "Given a movies dataset and a ratings data structure as returned by get-ratings build a matrix [num-movies X num-users] where a cell (i,j) represets the rating given to movie i by user j"
  [movies ratings]
  (let [num-movies (count movies)
        movieid->ymatrixid (nativeids->matrixids movies num-movies first)
        users-ids (into #{} (map first) (keys ratings))
        num-users (count users-ids)
        userid->ymatrixid (nativeids->matrixids users-ids num-users identity)
        ymatrix (m/mutable (m/zero-matrix num-movies num-users))]
    (doall (pmap (fn [[movie-idx _ _]]
                   (doall (map (fn [user-idx] ;; (prn "Rating for movie: "
                                 ;;      (ratings [user-idx movie-idx]))
                                 (m/mset! ymatrix
                                          (movieid->ymatrixid movie-idx)
                                          (userid->ymatrixid user-idx)
                                          (or (ratings [user-idx movie-idx]) 0.0)))
                               users-ids)))
                 movies))
    ;; (prn (m/mget ymatrix (movieid->ymatrixid 1391) (userid->ymatrixid 387)))
    ymatrix))

(defn create-rmatrix
  "Syntactic sugar for the matrix returned by create-ymatrix except that all non-zero values are now set to 1.0"
  [ymatrix]
  (pmap #(map (fn [x]
                (if (-> x zero?) x 1.0))
              %)
        ymatrix))

;; (def comedies (get-movies movies-csv :comedy))
;; (def ratings (get-ratings ratings-csv comedies))
 
;; (def ymatrix (create-ymatrix comedies ratings))
;; (def rmatrix (create-rmatrix ymatrix))

;; (count comedies)
;; (first ratings)

(defn title->movies
  "Basic search movies containing title-str within their title"
  [movies-list title-str]
  (let [title-str (str/lower-case title-str)]
    (filter #(->> %
                  second
                  str/lower-case
                  (re-find (re-pattern title-str)))
            movies-list)))

;; ----------------------------------------------
;; spec
(s/def ::csv-data (s/coll-of (s/coll-of (s/or :number number?
                                              :string string?))))

(s/def ::genre (apply hash-set (keys genres)))

(s/def ::ratings (s/coll-of (s/tuple (s/tuple number? number?) number?)
                            :kind map?))

(s/def ::movies-list (s/coll-of (s/tuple nat-int? string? string?)))

(s/fdef safe-read-string
  :args (s/cat :s string?)
  :ret (s/or :valid-clj any?
             :string string?))

(s/fdef read-number
  :args (s/cat :s string?)
  :ret (s/or :number number?
             :string string?))


(s/fdef read-lens-csv
  :args (s/cat :csv-file string?
               :k #{:movies :ratings})
  :ret ::csv-data)

(s/fdef count-entity
  :args (s/and (s/cat :entities-full (s/or :coll coll?
                                           :matrix m/matrix?))
               #(every? (fn [x] (-> x first number?))
                        (-> % :entities-full second)))
  :ret number?)

(s/fdef get-movies
  :args (s/cat :movies-csv string? ;;::csv-data
               :genre ::genre)
  :ret ::movies-list)

(s/fdef get-ratings
  :args (s/cat :ratings-csv string? ;;::csv-data
               :movies ::movies-list)
  :ret ::ratings)


(s/fdef nativeids->matrixids
  :args (s/and (s/cat :natives (s/coll-of coll?)
                      :natives-count number?
                      :selector-fn ifn?)
               #(= (:natives-count %) (-> % :natives count)))
  :ret map?
  :fn (s/and #(= (-> % :ret vals sort)
                 (-> % :args :natives-count range))
             
             #(= (-> % :ret keys sort)
                 (->> % :args :natives (map (-> % :args :selector-fn)) sort))))

(s/fdef matrixids->nativeids
  :args (s/and (s/cat :natives (s/coll-of coll?)
                      :natives-count number?
                      :selector-fn ifn?)
               #(= (:natives-count %) (-> % :natives count)))
  :ret map?
  :fn (s/and #(= (-> % :ret keys sort)
                 (-> % :args :natives-count range))
             
             #(= (-> % :ret vals sort)
                 (->> % :args :natives (map (-> % :args :selector-fn)) sort))))

(s/fdef create-ymatrix
  :args (s/cat :movies ::movies-list
               :ratings ::ratings)
  
  :ret m/matrix?
  :fn #(= (-> % :ret m/shape)
          [(-> % :args :movies count)
           (->> % :args :ratings keys (into #{} (map first)))]))

(s/fdef create-rmatrix
  :args (s/cat :ymatrix m/matrix?)
  :ret :recommender.particles/binary-matrix
  :fn #(m/same-shape? (-> % :args :ymatrix)
                      (:ret %)))

(s/fdef title->movies
  :args (s/cat :movies-list ::movies-list
               :title-str string?)
  :ret ::csv-data
  ;;:fn check that tuples in ret collection are found within initial movie-list
  )

;; (stest/instrument [`count-entity])

