import landbosse
from landbosse.landbosse_api.run import run_landbosse
import xlwings as xw

jedi_inputs = dict()

sheets = dict()
sheets['input'] = "STEP 1 - Project Information"
sheets['output'] = "STEP 2 - Cost Information"


def run():
     wb = xw.Book.caller()
     jedi_inputs['project_id'] = wb.sheets[sheets['input']].range('C5').value
     jedi_inputs['num_turbines'] = wb.sheets[sheets['input']].range('C10').value
     jedi_inputs['turbine_rating_MW'] = wb.sheets[sheets['input']].range('C13').value
     jedi_inputs['hub_height_meters'] = wb.sheets[sheets['input']].range('C14').value
     jedi_inputs['rotor_diameter_m'] = wb.sheets[sheets['input']].range('C15').value
     jedi_inputs['turbine_spacing_rotor_diameters'] = wb.sheets[sheets['input']].range('C16').value
     jedi_inputs['row_spacing_rotor_diameters'] = wb.sheets[sheets['input']].range('C17').value
     jedi_inputs['row_spacing_rotor_diameters'] = wb.sheets[sheets['input']].range('C22').value
     jedi_inputs['wind_shear_exponent'] = wb.sheets[sheets['input']].range('G5').value
     jedi_inputs['gust_velocity_m_per_s'] = wb.sheets[sheets['input']].range('G6').value
     jedi_inputs['num_access_roads'] = wb.sheets[sheets['input']].range('G7').value
     jedi_inputs['num_hwy_permits'] = wb.sheets[sheets['input']].range('G8').value
     jedi_inputs['fraction_new_roads'] = wb.sheets[sheets['input']].range('G10').value
     jedi_inputs['road_width_ft'] = wb.sheets[sheets['input']].range('G11').value
     jedi_inputs['road_thickness'] = wb.sheets[sheets['input']].range('G12').value
     jedi_inputs['fuel_usd_per_gal'] = wb.sheets[sheets['input']].range('G13').value
     jedi_inputs['road_quality'] = wb.sheets[sheets['input']].range('G14').value
     jedi_inputs['road_length_adder_m'] = wb.sheets[sheets['input']].range('G15').value
     jedi_inputs['crane_width'] = wb.sheets[sheets['input']].range('G16').value
     jedi_inputs['distance_to_interconnect_mi'] = wb.sheets[sheets['input']].range('G19').value
     jedi_inputs['interconnect_voltage_kV'] = wb.sheets[sheets['input']].range('G20').value
     jedi_inputs['line_frequency_hz'] = wb.sheets[sheets['input']].range('G21').value
     jedi_inputs['new_switchyard'] = wb.sheets[sheets['input']].range('G22').value
     jedi_inputs['distance_to_grid_connection_km'] = wb.sheets[sheets['input']].range('G23').value
     jedi_inputs['depth'] = wb.sheets[sheets['input']].range('G26').value
     jedi_inputs['rated_thrust_N'] = wb.sheets[sheets['input']].range('G27').value
     jedi_inputs['bearing_pressure_n_m2'] = wb.sheets[sheets['input']].range('G28').value
     jedi_inputs['critical_height_non_erection_wind_delays_m'] = wb.sheets[sheets['input']].range('G29').value
     jedi_inputs['critical_speed_non_erection_wind_delays_m_per_s'] = wb.sheets[sheets['input']].range('G30').value
     development_labor_usd_kw = float(wb.sheets[sheets['input']].range('G33').value)
     jedi_inputs['development_labor_cost_usd'] = development_labor_usd_kw * jedi_inputs['turbine_rating_MW'] * jedi_inputs['num_turbines'] * 1000

     BOS_results = run_landbosse(jedi_inputs)



     wb.sheets[sheets['output']].range('D5').value = jedi_inputs['development_labor_cost_usd']
     wb.sheets[sheets['output']].range('D6').value = 0
     wb.sheets[sheets['output']].range('D7').value = 0
     wb.sheets[sheets['output']].range('D8').value = 0
     wb.sheets[sheets['output']].range('D9').value = 0
     wb.sheets[sheets['output']].range('D10').value = 0


     wb.sheets[sheets['output']].range('D11').value = BOS_results['total_sitepreparation_cost']
     wb.sheets[sheets['output']].range('D12').value = BOS_results['sitepreparation_material_usd']
     wb.sheets[sheets['output']].range('D13').value = BOS_results['sitepreparation_equipment_rental_usd']
     wb.sheets[sheets['output']].range('D14').value = BOS_results['sitepreparation_labor_usd']
     wb.sheets[sheets['output']].range('D15').value = BOS_results['sitepreparation_mobilization_usd']
     wb.sheets[sheets['output']].range('D16').value = BOS_results['sitepreparation_other_usd']


     wb.sheets[sheets['output']].range('D17').value = BOS_results['total_foundation_cost']
     wb.sheets[sheets['output']].range('D18').value = BOS_results['foundation_equipment_rental_usd']
     wb.sheets[sheets['output']].range('D19').value = BOS_results['foundation_labor_usd']
     wb.sheets[sheets['output']].range('D20').value = BOS_results['foundation_material_usd']
     wb.sheets[sheets['output']].range('D21').value = BOS_results['foundation_mobilization_usd']


     wb.sheets[sheets['output']].range('D22').value = BOS_results['total_erection_cost']
     wb.sheets[sheets['output']].range('D23').value = BOS_results['erection_equipment_rental_usd']
     wb.sheets[sheets['output']].range('D24').value = BOS_results['erection_fuel_usd']
     wb.sheets[sheets['output']].range('D25').value = BOS_results['erection_labor_usd']
     wb.sheets[sheets['output']].range('D26').value = BOS_results['erection_material_usd']
     wb.sheets[sheets['output']].range('D27').value = BOS_results['erection_mobilization_usd']
     wb.sheets[sheets['output']].range('D28').value = BOS_results['erection_other_usd']


     wb.sheets[sheets['output']].range('D29').value = BOS_results['total_collection_cost']
     wb.sheets[sheets['output']].range('D30').value = BOS_results['collection_equipment_rental_usd']
     wb.sheets[sheets['output']].range('D31').value = BOS_results['collection_labor_usd']
     wb.sheets[sheets['output']].range('D32').value = BOS_results['collection_material_usd']
     wb.sheets[sheets['output']].range('D33').value = BOS_results['collection_mobilization_usd']


     wb.sheets[sheets['output']].range('D34').value = BOS_results['total_gridconnection_cost']


     wb.sheets[sheets['output']].range('D35').value = BOS_results['total_substation_cost']


     wb.sheets[sheets['output']].range('D36').value = BOS_results['total_management_cost']
     wb.sheets[sheets['output']].range('D37').value = BOS_results['insurance_usd']
     wb.sheets[sheets['output']].range('D38').value = BOS_results['construction_permitting_usd']
     wb.sheets[sheets['output']].range('D39').value = BOS_results['project_management_usd']
     wb.sheets[sheets['output']].range('D40').value = BOS_results['bonding_usd']


     wb.sheets[sheets['output']].range('D41').value = BOS_results['engineering_usd']
     wb.sheets[sheets['output']].range('D42').value = BOS_results['site_facility_usd']





