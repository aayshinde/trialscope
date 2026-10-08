-- Resolved sponsor per trial (entity resolution output from trialscope.entity -> raw.sponsor_map)
select
    s.nct_id,
    coalesce(m.sponsor_key, lower(s.lead_sponsor_name))  as sponsor_key,
    coalesce(m.sponsor_name, s.lead_sponsor_name)        as sponsor_name
from {{ ref('stg_studies') }} s
left join {{ source('raw', 'sponsor_map') }} m on m.raw_name = s.lead_sponsor_name
