-- =========================================================================
-- BigQuery VECTOR_SEARCH Example Queries (2026 Modern User Similarity)
-- Project: marine-access-331406 | Dataset: pci_vector_search | Table: users_10k
-- =========================================================================

-- 1. Check Vector Index Status and Coverage
SELECT
  table_name,
  index_name,
  index_status,
  coverage_percentage,
  last_refresh_time,
  distance_type,
  index_type
FROM
  `marine-access-331406.pci_vector_search.INFORMATION_SCHEMA.VECTOR_INDEXES`;

-- 2. Approximate Search with IVF/TreeAH Vector Index (Fastest)
SELECT
  query.user_id AS target_user,
  base.user_id AS similar_user,
  base.segment_name,
  base.profile_text,
  (1 - distance) AS cosine_similarity,
  distance
FROM
  VECTOR_SEARCH(
    TABLE `marine-access-331406.pci_vector_search.users_10k`,
    'embedding',
    (SELECT user_id, embedding FROM `marine-access-331406.pci_vector_search.users_10k` WHERE user_id = 'usr_000001'),
    top_k => 10,
    distance_type => 'COSINE',
    options => '{"use_brute_force": false}'
  )
WHERE base.user_id != query.user_id
ORDER BY distance ASC;

-- 3. Exact Search (Brute Force) without index
SELECT
  query.user_id AS target_user,
  base.user_id AS similar_user,
  base.segment_name,
  base.profile_text,
  (1 - distance) AS cosine_similarity,
  distance
FROM
  VECTOR_SEARCH(
    TABLE `marine-access-331406.pci_vector_search.users_10k`,
    'embedding',
    (SELECT user_id, embedding FROM `marine-access-331406.pci_vector_search.users_10k` WHERE user_id = 'usr_000001'),
    top_k => 10,
    distance_type => 'COSINE',
    options => '{"use_brute_force": true}'
  )
WHERE base.user_id != query.user_id
ORDER BY distance ASC;
