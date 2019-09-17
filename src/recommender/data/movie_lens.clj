(ns recommender.data.movie-lens
  (:require [clojure.data.csv :as csv]
            [clojure.core.matrix :as m]
            [clojure.java.io :as io]
            [clojure.string :as str]))

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

(defn safe-read-string [s]
  (try (read-string s)
       (catch Exception e s)))

(defn read-number [s]
  (let [n? (safe-read-string s)]
    (if (number? n?) n? s)))

(defmulti read-lens-csv-row (fn [_ k] k))

(defmethod read-lens-csv-row :movies
  [[id title genre :as csv-row] _]
  (vector (read-number id) title genre))

(defmethod read-lens-csv-row :ratings
  [[uid movid rating timestamp :as csv-row] _]
  (mapv read-number csv-row))

(defn read-lens-csv [csv-file k]
  (pmap #(read-lens-csv-row % k)
        (drop 1 (with-open [reader (io/reader csv-file)]
                  (doall (csv/read-csv reader))))))

(defn count-entity [entities-full]
  (apply max (map first entities-full)))
;; (def num-movies (count-entity (read-lens-csv movies-csv)))

(defn- update-ymatrix [transient-result [user-id movie-id rating :as rating-line]]
  (assoc! transient-result
          (dec movie-id)
          (assoc (nth transient-result (dec movie-id)) (dec user-id) rating)))

(defn create-ymatrix [movies ratings]
  (let [num-movies (count movies)
        max-movie-id (count-entity movies)
        num-users (count-entity ratings)
        zero-yrow (zipmap (range num-users) (repeat num-users 0.0))
        ;; ymatrix (loop [ratings ratings-full
        ;;                 result (transient (into [] (repeat num-movies {})))]
        ;;            (if (-> ratings seq not)
        ;;              (persistent! result)
        ;;              (recur (rest ratings)
        ;;                     (update-ymatrix result (first ratings)))))
        ]
        
    ;;ymatrix
    (pmap #(->> % (merge zero-yrow) sort vals)
          (loop [ratings ratings
                 result (transient (into [] (repeat num-movies {})))]
            (if (-> ratings seq not)
              (persistent! result)
              (recur (rest ratings)
                     (update-ymatrix result (first ratings))))))
    
    ))

(defn get-movies [movies-csv genre]
  (filter #(->> % last (re-find (-> genres genre re-pattern)))
          (read-lens-csv movies-csv :movies)))

(defn get-ratings [ratings-csv movies]
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

(defn nativeids->matrixids [natives natives-count selector-fn]
  (zipmap (sort (map selector-fn natives))
          (range natives-count)))

(defn matrixids->nativeids [natives natives-count selector-fn]
  (zipmap (range natives-count)
          (sort (map selector-fn natives))))

(defn create-ymatrix [movies ratings]
  (let [num-movies (count movies)
        movieid->ymatrixid (nativeids->matrixids movies num-movies first)
        users-ids (into #{} (map first (keys ratings)))
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

(defn create-rmatrix [ymatrix]
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

(defn title->movie [movies-list title-str]
  (let [title-str (str/lower-case title-str)]
    (filter #(->> %
                  second
                  str/lower-case
                  (re-find (re-pattern title-str)))
            movies-list)))
