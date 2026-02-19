# RobDP: PostgreSQL 16 with Built-in Robustness

RobDP extends PostgreSQL’s DP optimizer with minimal changes. We expose three configurable hooks — local objective, diversifying objective, and final objective — for local pruning, diversification, and final plan selection, with an optional two-pass DP for lower envelope exploration.

To reduce planning overhead for complex objectives, RobDP also integrates with parametric query optimization (PQO) for efficient candidate precomputation and runtime selection. It serves as a general testbed for robust optimization research, allowing users to plug in custom objectives and diversification strategies.

This project provides a streamlined setup for PostgreSQL 16 with built-in robustness. We include the balsa dataloader to initialize the IMDb database, along with GUC parameters and a concrete query example to get you started quickly.

---

## 1. Clone and Install RobDP

```bash
# In your working directory
git clone https://github.com/Hap-Hugh/PostgreSQL-16-Robust
cd PostgreSQL-16-Robust

# Configure with optional installation path
./configure --prefix=/usr/local/pgsql/robdp

# Add --without-icu if ICU library problems occur
./configure --prefix=/usr/local/pgsql/robdp --without-icu

# Build and install
make -j
make install
````

### Set environment variables

```bash
export PGHOME=/usr/local/pgsql/robdp   # use your actual prefix path
export PATH="$PGHOME/bin:$PATH"
```

---

## 2. Initialize and Start the PostgreSQL Database

```bash
# Clean up previous instance if exists
rm -rf ./imdb

# Initialize database
pg_ctl -D ./imdb initdb

# Use the provided PostgreSQL config
cp ./PG16/postgresql.conf ./imdb

# Start the database
pg_ctl -D ./imdb -l ./dblogs/logfile start
```

---

## 3. Download and Load IMDb Data (via the [Balsa Loader](https://github.com/balsa-project/balsa))

```bash
# In your working directory
cd ./PostgreSQL-16-Robust

# Download IMDb dataset
wget https://event.cwi.nl/da/job/imdb.tgz

# Extract into a clean datasource directory
mkdir ./datasource
cd ./datasource
tar zxvf ../imdb.tgz

# Run the Python script to prepend IMDb header
python3 ./prepend_imdb_headers.py

# Run the loader script
bash ./load_job_postgres.sh [full path to your datasource directory]
```

---

## 4. GUC Parameters

RobDP can be configured via the following PostgreSQL GUC parameters.

### File path configuration

| GUC                  | Type / Range | Default | Description                                                                                           |
| -------------------- | -----------: | ------: | ----------------------------------------------------------------------------------------------------- |
| `error_profile_path` |         path |       – | Directory for storing error profile files. Ensure proper filesystem permissions.                      |
| `score_filename`     |     filename |       – | Output filename for per-query robustness score. Target directory must exist and be readable/writable. |

### Error profile sampling configuration

| GUC                          |    Type / Range | Default | Description                                     |
| ---------------------------- | --------------: | ------: | ----------------------------------------------- |
| `error_sample_count`         |      int (4–64) |      20 | Number of samples drawn from the error profile. |
| `error_bin_count`            |       int (1–8) |       1 | Number of bins for splitting error profiles.    |
| `error_sample_kde_bandwidth` | float (0.0–2.0) |     0.5 | Bandwidth used for KDE-based sampling.          |
| `error_sample_seed`          |             int |       – | Random seed used for drawing error samples.     |

### Path objective configuration

| GUC            | Type / Range | Default | Description                                             |
| -------------- | -----------: | ------: | ------------------------------------------------------- |
| `main_objective_id` |   int (0–16) |       0 | Objective ID for ObjLocal (local pruning).              |
| `retain_strategy_id`   |   int (0–16) |       0 | Objective ID for Diversify (candidate diversification). |
| `final_score_id` |   int (0–16) |       0 | Objective ID for Obj (final plan selection).            |

### Path limit configuration

| GUC                    | Type / Range | Default | Description                                    |
| ---------------------- | -----------: | ------: | ---------------------------------------------- |
| `add_path_limit` |   int (1–64) |       1 | Maximum number of paths retained by ObjLocal.  |
| `retain_path_limit`   |   int (0–64) |       1 | Maximum number of paths retained by Diversify. |

### Objective IDs

| ID | Meaning                                      |
| -: | -------------------------------------------- |
|  0 | Quantile[Penalty] (alpha-quantile of penalty) |
|  1 | E[Penalty]                                   |
|  3 | E[Cost]                                      |
|  4 | E[Startup Cost]                                     |
|  5 | Quantile[Startup Cost] (alpha-quantile of startup cost)           |
|  6 | ProbNoPenalty                                     |
|  7 | Rand                                         |
| 11 | E[Penalty] + lambda * Stdev                    |
| 13 | Similarity                                   |
| 14 | Set                                          |
| 16 | E[Cost]-Join                                 |

---

## 5. Example: Query 10 from Template 10 in JOB's Carver Workload

This section presents a concrete query example from JOB.

In this example:

* **ObjLocal** is configured as `E[Penalty]` to retain one path.
* **Diversify** is configured as `Similarity` with a limit of 8 paths per join relation (i.e., `<Similarity, 8>`).
* **Obj** (final objective) is set to `E[Penalty]`.

For error sampling, we use:

* `N = 20` samples
* `8` bins
* KDE bandwidth = `0.1`

These settings are controlled via PostgreSQL GUC parameters as shown below.

---

### GUC Parameter Settings

```sql
SET error_profile_path = 'replace with your own error profile directory';
SET enable_rows_dist = on;
SET error_sample_count = 20;
SET error_bin_count = 8;
SET error_sample_kde_bandwidth = 0.1;

SET main_objective_id = 1;      -- E[Penalty]
SET retain_strategy_id = 13;       -- Similarity
SET final_score_id = 1;      -- E[Penalty]
SET add_path_limit = 1;
SET retain_path_limit = 8;
```

---

### Query Overview

The example query is **Query 10 of Template 10** from the JOB Carver workload:

```sql
SELECT min(chn.name) AS uncredited_voiced_character,
       min(t.title) AS russian_movie
FROM char_name AS chn,
     cast_info AS ci,
     company_name AS cn,
     company_type AS ct,
     movie_companies AS mc,
     role_type AS rt,
     title AS t
WHERE t.id = mc.movie_id
  AND t.id = ci.movie_id
  AND ci.movie_id = mc.movie_id
  AND chn.id = ci.person_role_id
  AND rt.id = ci.role_id
  AND cn.id = mc.company_id
  AND ct.id = mc.company_type_id
  AND ci.note LIKE '%d%'
  AND ci.note LIKE '%archive%'
  AND cn.country_code = '[us]'
  AND rt.role = 'actor'
  AND t.production_year > 2002;
```

The result should be similar to the following plan:
```text
 Finalize Aggregate  (cost=171231.45..171231.46 rows=1 width=64)
   ->  Gather  (cost=171231.12..171231.43 rows=3 width=64)
         Workers Planned: 3
         ->  Partial Aggregate  (cost=170231.12..170231.13 rows=1 width=64)
               ->  Hash Join  (cost=115544.05..170220.28 rows=2168 width=33)
                     Hash Cond: (mc.company_type_id = ct.id)
                     ->  Nested Loop  (cost=115542.96..170211.24 rows=1472 width=37)
                           ->  Nested Loop  (cost=115542.53..168500.27 rows=3317 width=28)
                                 ->  Hash Join  (cost=115542.10..148066.00 rows=26322 width=16)
                                       Hash Cond: (mc.movie_id = ci.movie_id)
                                       ->  Parallel Hash Join  (cost=5333.06..34749.00 rows=293296 width=8)
                                             Hash Cond: (mc.company_id = cn.id)
                                             ->  Parallel Seq Scan on movie_companies mc  (cost=0.00..27206.55 rows=841655 width=12)
                                             ->  Parallel Hash  (cost=4722.92..4722.92 rows=48812 width=4)
                                                   ->  Parallel Seq Scan on company_name cn  (cost=0.00..4722.92 rows=48812 width=4)
                                                         Filter: ((country_code)::text = '[us]'::text)
                                       ->  Hash  (cost=108423.88..108423.88 rows=142813 width=8)
                                             ->  Nested Loop  (cost=0.44..108423.88 rows=142813 width=8)
                                                   ->  Seq Scan on role_type rt  (cost=0.00..1.15 rows=1 width=4)
                                                         Filter: ((role)::text = 'actor'::text)
                                                   ->  Index Scan using role_id_cast_info on cast_info ci  (cost=0.44..108342.59 rows=8014 width=12)
                                                         Index Cond: (role_id = rt.id)
                                                         Filter: ((note ~~ '%d%'::text) AND (note ~~ '%archive%'::text))
                                 ->  Index Scan using char_name_pkey on char_name chn  (cost=0.43..0.78 rows=1 width=20)
                                       Index Cond: (id = ci.person_role_id)
                           ->  Index Scan using title_pkey on title t  (cost=0.43..0.52 rows=1 width=21)
                                 Index Cond: (id = mc.movie_id)
                                 Filter: (production_year > 2002)
                     ->  Hash  (cost=1.04..1.04 rows=4 width=4)
                           ->  Seq Scan on company_type ct  (cost=0.00..1.04 rows=4 width=4)
(30 rows)
```

## 6. Support for Multi-Block Queries
- Use the `dsb` branch for DSB SPJ queries.
- Use the `dsb-multi` branch for DSB multi-block queries.

---

## Notes

* Make sure the port (`PGPORT`) matches your PostgreSQL instance.
* The default DB name created is `imdbload`.
* Please copy the `PostgreSQL-16-Robust`'s config file `./PostgreSQL-16-Robust/postgresql.conf` to the `imdb` directory.

## Reference 

We welcome all forms of discussion and collaboration for future research and are eager to help with plan selection in your project. If you find this repository valuable, please consider citing our work ...
