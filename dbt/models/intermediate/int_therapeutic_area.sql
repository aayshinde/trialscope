-- Primary therapeutic area from the MeSH browse branches ClinicalTrials.gov derives for each trial.
-- Generic branches are de-prioritised; 'Rare Diseases' becomes a separate flag.
with b as (
    select
        nct_id, branch_name,
        case branch_name
            when 'Neoplasms' then 1
            when 'Heart and Blood Diseases' then 2
            when 'Nervous System Diseases' then 3
            when 'Behaviors and Mental Disorders' then 4
            when 'Infections' then 5
            when 'Nutritional and Metabolic Diseases' then 6
            when 'Immune System Diseases' then 7
            when 'Respiratory Tract (Lung and Bronchial) Diseases' then 8
            when 'Digestive System Diseases' then 9
            when 'Skin and Connective Tissue Diseases' then 10
            when 'Muscle, Bone, and Cartilage Diseases' then 11
            when 'Urinary Tract, Sexual Organs, and Pregnancy Conditions' then 12
            when 'Symptoms and General Pathology' then 90
            when 'All Conditions' then 99
            when 'Rare Diseases' then 100
            else 50
        end as priority
    from {{ ref('stg_browse_branches') }}
),
ranked as (
    select nct_id, branch_name,
           row_number() over (partition by nct_id order by priority, branch_name) as rn
    from b where priority < 99
)
select
    s.nct_id,
    coalesce(r.branch_name, 'Other / unspecified')                                      as therapeutic_area,
    coalesce(bool_or(b.branch_name = 'Rare Diseases'), false)                           as is_rare_disease,
    count(distinct b.branch_name) filter (where b.priority < 99)                        as n_branches
from {{ ref('stg_studies') }} s
left join ranked r on r.nct_id = s.nct_id and r.rn = 1
left join b on b.nct_id = s.nct_id
group by 1, 2
