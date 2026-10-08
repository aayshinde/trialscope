-- One row per interventional trial: attributes known at start, resolved sponsor, therapeutic area,
-- leakage-safe sponsor track record, outcomes/survival times and SCD2 history metrics.
select
    s.nct_id, s.brief_title, s.overall_status, s.status_group, s.why_stopped,
    s.start_date, s.start_year, s.primary_completion_date, s.primary_completion_type, s.first_submit_date,
    s.last_update_date,
    sp.sponsor_key, sp.sponsor_name, s.sponsor_class, s.n_collaborators,
    s.phase, s.phase_group, s.allocation, s.intervention_model, s.primary_purpose, s.masking,
    s.enrollment_count, s.enrollment_type, s.n_arms,
    s.healthy_volunteers, s.sex, s.min_age_years, s.max_age_years,
    s.n_locations, s.n_countries, s.has_results,
    ta.therapeutic_area, ta.is_rare_disease, ta.n_branches,
    tr.sponsor_prior_closed, tr.sponsor_prior_terminated, tr.sponsor_prior_started, tr.sponsor_prior_term_rate,
    o.snapshot_date, o.is_resolved, o.terminated_flag, o.duration_days, o.completion_event,
    o.planned_primary_completion, o.planned_duration_days, o.is_overdue,
    h.n_versions, h.n_status_changes, h.n_completion_slips, h.first_observed_at, h.first_observed_status,
    h.first_planned_completion, h.first_planned_enrollment,
    case when h.first_planned_enrollment > 0 and s.enrollment_type = 'ACTUAL'
         then s.enrollment_count::double / h.first_planned_enrollment end      as enrollment_vs_plan,
    coalesce(iv.n_drug_interventions, 0)                                     as n_drug_interventions,
    coalesce(rf.n_result_publications, 0)                                    as n_result_publications
from {{ ref('stg_studies') }} s
join {{ ref('int_sponsors') }} sp using (nct_id)
left join {{ ref('int_therapeutic_area') }} ta using (nct_id)
left join {{ ref('int_sponsor_track_record') }} tr using (nct_id)
left join {{ ref('int_trial_outcomes') }} o using (nct_id)
left join {{ ref('int_study_history') }} h using (nct_id)
left join (select nct_id, count(*) as n_drug_interventions from {{ ref('stg_interventions') }}
           where intervention_type = 'DRUG' group by 1) iv using (nct_id)
left join (select nct_id, count(*) as n_result_publications from {{ ref('stg_references') }}
           where ref_type = 'RESULT' group by 1) rf using (nct_id)
