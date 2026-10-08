{# Use the configured schema name as-is (staging, intermediate, marts, snapshots) instead of main_staging etc. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {{ custom_schema_name | trim if custom_schema_name is not none else target.schema }}
{%- endmacro %}
