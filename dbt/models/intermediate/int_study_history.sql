-- What the SCD2 snapshot knows about each trial's evolution.
-- planned_* come from the FIRST version we observed while the value was still an estimate,
-- i.e. genuine at-registration plans for trials the pipeline has tracked since early on.
with v as (select * from {{ ref('snap_studies') }}),
first_est as (
    select nct_id,
           arg_min(primary_completion_date, dbt_valid_from) filter (where primary_completion_type = 'ESTIMATED') as first_planned_completion,
           arg_min(enrollment_count, dbt_valid_from) filter (where enrollment_type = 'ESTIMATED')              as first_planned_enrollment,
           min(dbt_valid_from)                                                                                  as first_observed_at,
           arg_min(overall_status, dbt_valid_from)                                                              as first_observed_status
    from v group by 1
),
changes as (
    select nct_id,
           count(*)                                                                          as n_versions,
           count(*) filter (where overall_status is distinct from prev_status)  - 1          as n_status_changes,
           count(*) filter (where primary_completion_type = 'ESTIMATED' and prev_type = 'ESTIMATED'
                            and primary_completion_date > prev_pc)                           as n_completion_slips
    from (
        select *, lag(overall_status) over w as prev_status, lag(primary_completion_date) over w as prev_pc,
               lag(primary_completion_type) over w as prev_type
        from v window w as (partition by nct_id order by dbt_valid_from)
    ) x group by 1
)
select f.*, c.n_versions, greatest(c.n_status_changes, 0) as n_status_changes, c.n_completion_slips
from first_est f join changes c using (nct_id)
