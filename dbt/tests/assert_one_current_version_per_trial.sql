select nct_id, count(*) as n_current
from {{ ref('snap_studies') }}
where dbt_valid_to is null
group by 1
having count(*) != 1
