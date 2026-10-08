select distinct nct_id, pmid, ref_type from {{ source('raw', 'references') }} where pmid is not null
