select distinct nct_id, abbrev, name as branch_name from {{ source('raw', 'browse_branches') }} where name is not null
