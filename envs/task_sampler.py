"""
Task sampler over admitted 2023-family CityLearn anchors, each task conformed to a FIXED
(obs_dim=52, action_dim=9) layout.

Admitted anchors: all 8 `citylearn_challenge_2023_*` schemas share the same building template
(dhw_storage/electrical_storage/cooling_device actions; day_type/hour/outdoor_temp+forecasts/
solar+forecasts/carbon_intensity/pricing+forecasts shared obs; indoor_temp/non_shiftable_load/
solar_generation/dhw_storage_soc/electrical_storage_soc/net_electricity_consumption/
cooling_demand/dhw_demand/occupant_count/indoor_temp_set_point/power_outage per-building obs).
The 2022/2021/baeda_3dem families are EXCLUDED: they lack cooling_device entirely (different
building template), not just a renaming -- no padding or truncation is used to force them to fit.

`load_conformed_schema` is the shared entry point used by `combined_task_sampler.py` to build the
current task distribution (see that module for the active TRAIN/HELD-OUT task definitions).
"""
import copy
import json
import os

CHESCA_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'oracle', 'chesca_repo')

REQUIRED_SHARED_OBS = [
    'day_type', 'hour',
    'outdoor_dry_bulb_temperature', 'outdoor_dry_bulb_temperature_predicted_6h',
    'outdoor_dry_bulb_temperature_predicted_12h', 'outdoor_dry_bulb_temperature_predicted_24h',
    'diffuse_solar_irradiance', 'diffuse_solar_irradiance_predicted_6h',
    'diffuse_solar_irradiance_predicted_12h', 'diffuse_solar_irradiance_predicted_24h',
    'direct_solar_irradiance', 'direct_solar_irradiance_predicted_6h',
    'direct_solar_irradiance_predicted_12h', 'direct_solar_irradiance_predicted_24h',
    'carbon_intensity',
    'electricity_pricing', 'electricity_pricing_predicted_6h',
    'electricity_pricing_predicted_12h', 'electricity_pricing_predicted_24h',
]
REQUIRED_PER_BUILDING_OBS = [
    'indoor_dry_bulb_temperature', 'non_shiftable_load', 'solar_generation', 'dhw_storage_soc',
    'electrical_storage_soc', 'net_electricity_consumption', 'cooling_demand', 'dhw_demand',
    'occupant_count', 'indoor_dry_bulb_temperature_set_point', 'power_outage',
]
REQUIRED_OBS = set(REQUIRED_SHARED_OBS) | set(REQUIRED_PER_BUILDING_OBS)
REQUIRED_ACTIONS = {'dhw_storage', 'electrical_storage', 'cooling_device'}


def load_conformed_schema(rel_path, building_subset=None):
    """Load a schema and conform it to EXACTLY the required (52, 9) layout:
    - top-level observations/actions: active=True for exactly the required fields, False for
      everything else.
    - per building: clear inactive_observations/inactive_actions overrides (rely purely on the
      top-level active flags), and restrict to `building_subset` indices if given (else all).
    - force central_agent=True.
    """
    full_path = os.path.join(CHESCA_REPO, rel_path)
    with open(full_path) as f:
        schema = json.load(f)
    schema = copy.deepcopy(schema)

    for field, meta in schema['observations'].items():
        meta['active'] = field in REQUIRED_OBS
    for field, meta in schema['actions'].items():
        meta['active'] = field in REQUIRED_ACTIONS

    building_names = list(schema['buildings'].keys())
    if building_subset is not None:
        keep_names = [building_names[i] for i in building_subset]
    else:
        keep_names = building_names
    assert len(keep_names) == 3, f"conformed task must have exactly 3 buildings, got {len(keep_names)}"

    schema['buildings'] = {name: schema['buildings'][name] for name in keep_names}
    for b in schema['buildings'].values():
        b['inactive_observations'] = []
        b['inactive_actions'] = []

    schema['central_agent'] = True
    schema['root_directory'] = os.path.join(CHESCA_REPO, os.path.dirname(rel_path))
    return schema
