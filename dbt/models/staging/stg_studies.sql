-- One row per trial (current version), typed and lightly derived. No business logic beyond grouping.
select
    nct_id,
    brief_title,
    overall_status,
    case
        when overall_status = 'COMPLETED' then 'completed'
        when overall_status = 'TERMINATED' then 'terminated'
        when overall_status = 'WITHDRAWN' then 'withdrawn'
        when overall_status = 'SUSPENDED' then 'suspended'
        when overall_status in ('RECRUITING', 'NOT_YET_RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION') then 'active'
        when overall_status = 'UNKNOWN' then 'unknown'
        else 'other'
    end                                             as status_group,
    nullif(trim(why_stopped), '')                   as why_stopped,
    cast(start_date as date)                        as start_date,
    start_date_type,
    year(cast(start_date as date))                  as start_year,
    cast(primary_completion_date as date)           as primary_completion_date,
    primary_completion_type,
    cast(completion_date as date)                   as completion_date,
    cast(first_submit_date as date)                 as first_submit_date,
    cast(last_update_date as date)                  as last_update_date,
    lead_sponsor_name,
    coalesce(lead_sponsor_class, 'UNKNOWN')         as sponsor_class,
    n_collaborators,
    phase,
    case
        when phase in ('EARLY_PHASE1', 'PHASE1') then 'Phase 1'
        when phase = 'PHASE1|PHASE2' then 'Phase 1/2'
        when phase = 'PHASE2' then 'Phase 2'
        when phase = 'PHASE2|PHASE3' then 'Phase 2/3'
        when phase = 'PHASE3' then 'Phase 3'
        when phase = 'PHASE4' then 'Phase 4'
        else 'Not applicable'
    end                                             as phase_group,
    coalesce(allocation, 'NA')                      as allocation,
    coalesce(intervention_model, 'NA')              as intervention_model,
    coalesce(primary_purpose, 'NA')                 as primary_purpose,
    coalesce(masking, 'NONE')                       as masking,
    cast(enrollment_count as integer)               as enrollment_count,
    enrollment_type,
    n_arms,
    eligibility_criteria,
    healthy_volunteers,
    sex,
    min_age_years,
    max_age_years,
    n_locations,
    n_countries,
    countries,
    has_results,
    loaded_at
from {{ source('raw', 'studies') }}
where study_type = 'INTERVENTIONAL' or study_type is null
