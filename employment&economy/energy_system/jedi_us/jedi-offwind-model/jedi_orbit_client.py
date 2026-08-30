

import os
import sys
import json
import argparse
from copy import deepcopy
from typing import Union, Any, Tuple, List
from pathlib import Path
from operator import getitem
from functools import reduce, partial
from collections import defaultdict
from distutils.dir_util import copy_tree

import yaml
import pandas as pd
import xlwings as xw
from ORBIT import ProjectManager
from ORBIT.core.library import initialize_library
from yaml import Dumper



initialize_library(None)
LIBRARY =  Path(os.environ["DATA_LIBRARY"])

WEATHER = Path(__file__).parent.absolute() / "jedi_weather_profiles"


REGION_MAP = {
    state: "Gulf of Maine"
    for state in ["Maine", "New Hampshire", "Massachusetts", "Gulf of Maine"]
}
REGION_MAP.update({
    state: "North Atlantic"
    for state in [
        "Massachusetts", "Rhode Island", "Connecticut", "North Atlantic",
        "Quebec", "New Brunswick", "Nova Scotia", "Prince Edward Island",
        "Newfoundland and Labrador",
    ]
})
REGION_MAP.update({
    state: "Mid-Atlantic"
    for state in [
        "New Jersey", "Delaware", "Maryland", "New York - Atlantic Coast",
        "Mid-Atlantic", "Atlantic Coast", "United States", "Define Region"
    ]
})
REGION_MAP.update({
    state: "South Atlantic"
    for state in ["Virginia", "North Carolina", "South Carolina", "Georgia", "South Atlantic"]
})
REGION_MAP.update({
    state: "Gulf of Mexico"
    for state in ["Texas", "Louisiana", "Alabama", "Florida", "Mississippi", "Gulf of Mexico"]
})
REGION_MAP.update({
    state: "Great Lakes"
    for state in [
        "Illinois", "Indiana", "Michigan", "Minnesota", "Ohio", "Wisconsin",
        "New York - Great Lakes", "Pennsylvania", "Great Lakes",
        "Ontario",
    ]
})
REGION_MAP.update({
    state: "West Coast" for state in [
        "Oregon", "California", "Washington", "West Coast", "British Columbia"
    ]
})
REGION_MAP.update({"Hawaii": "Hawaii"})


def _load_weather(user_input: str) -> pd.DataFrame:
    user_input = user_input.replace(" [Region]", "")
    region = REGION_MAP[user_input].lower().replace(" ", "_").replace("-", "_")
    weather_file = str(WEATHER / f"{region}_weather_2009_2020.csv")
    weather = pd.read_csv(weather_file, parse_dates=["datetime"]).set_index(keys="datetime")
    return weather


def _nested_dict_set(dictionary: dict, key_list: List[str], value: Any) -> None:
    *first, last = key_list
    _nested = dictionary
    for key in first:
        _nested = _nested.setdefault(key, defaultdict(dict))
    _nested[last] = value

def read_excel_inputs(wb: xw.Book, configuration: dict) -> dict:
    input_sheet = wb.sheets["Project Data - ORBIT Inputs"]

    rng = input_sheet.range("B17").expand("down").address
    rng1, rng2 = rng.split(":")
    rng = f"{rng1}:{rng2.replace('B', 'D')}"
    categories_list = input_sheet.range(rng).value
    values_list = input_sheet.range(f"$E$17:{rng2.replace('$B', '$E')}").value

    for categories, val in zip(categories_list, values_list):
        categories = [cat for cat in categories if cat is not None]
        val = int(1e40) if val == "infinite" else val

        if isinstance(val, float):
            try:
                val = int(val) if val % 1 == 0 else val
            except TypeError:
                pass

        multi_category = [cat.strip() for cat in categories[0].split(",")]
        for category in multi_category:
            _categories = [category] + categories[1:]
            try:
                _nested_dict_set(configuration, _categories, val)
            except AttributeError:
                raise AttributeError

    return configuration


def tidy_cable_data(array_config: dict, export_config: dict) -> dict:
    array = deepcopy(array_config)
    cable1 = deepcopy(array["cable1"])
    cable2 = deepcopy(array["cable2"])
    cable1["name"] = cable1["name"].replace(" ", "_")
    cable2["name"] = cable2["name"].replace(" ", "_")
    if cable2["name"] == "None":
        cables = [cable1]
    else:
        cables = [cable1, cable2]

    cables = {c["name"]: c for c in cables}
    array_config["cables"] = cables
    del array_config["cable1"]
    del array_config["cable2"]

    export = deepcopy(export_config)
    cable = deepcopy(export["cables"])
    cable["name"] = cable["name"].replace(" ", "_")
    export_config["cables"] = {cable["name"]: cable}

    return array_config, export_config

def determine_phases(wb: xw.Book) -> List[str]:

    design_phases = [
        "ArraySystemDesign",
        "ExportSystemDesign",
        "OffshoreSubstationDesign",
    ]
    install_phases = [
        "TurbineInstallation",
        "ArrayCableInstallation",
        "ExportCableInstallation",
        "OffshoreSubstationInstallation",
    ]

    substructure = wb.sheets["Project Data - Step 1"].range("D53").value
    if substructure == "Monopile":
        design_phases[0:0] = ["MonopileDesign", "ScourProtectionDesign"]
        install_phases[0:0] = ["MonopileInstallation", "ScourProtectionInstallation"]
    elif substructure == "Spar":
        design_phases[0:0] = ["SparDesign", "MooringSystemDesign"]
        install_phases[0:0] = ["MooringSystemInstallation", "MooredSubInstallation"]
    elif substructure == "Semisubmersible":
        design_phases[0:0] = ["SemiSubmersibleDesign", "MooringSystemDesign"]
        install_phases[0:0] = ["MooredSubInstallation", "MooringSystemInstallation"]

    return design_phases, install_phases

def determine_vessels(config: dict) -> dict:
    if "MonopileInstallation" in config["install_phases"]:
        config["oss_install_vessel"] = config["heavy_lift_vessel"]
        config["OffshoreSubstationInstallation"]["feeder"] = config["heavy_feeder"]
    else:
        config["oss_install_vessel"] = config["floating_heavy_lift_vessel"]
        config["OffshoreSubstationInstallation"]["feeder"] = config["floating_barge"]
        config["mooring_install_vessel"] = config["support_vessel"]
        config["wtiv"] = config["floating_heavy_lift_vessel"]
        config["feeder"] = config["floating_barge"]

    return config

def write_outputs(wb: xw.Book, project: ProjectManager) -> None:
    outputs = wb.sheets["Project Costs - Step 2"]

    installation_logs = pd.DataFrame.from_dict(project.logs)


    monopile_materials_cost = 0.0
    scour_protection_materials_cost = 0.0
    spar_materials_cost = 0.0
    semisubmersible_materials_cost = 0.0
    mooring_materials_cost = 0.0
    monopile_action_cost = 0.0
    monopile_port_cost = 0.0
    spar_action_cost = 0.0
    spar_port_cost = 0.0
    semisubmersible_action_cost = 0.0
    semisubmersible_port_cost = 0.0
    mooring_action_cost = 0.0
    mooring_port_cost = 0.0
    scour_protection_action_cost = 0.0
    scour_protection_port_cost = 0.0

    if "MonopileDesign" in project.config["design_phases"]:
        monopile_materials_cost = project.phases["MonopileDesign"].total_cost
        scour_protection_materials_cost = project.phases["ScourProtectionDesign"].total_cost
        monopile_installation_capex = project.phases["MonopileInstallation"].installation_capex
        monopile_port_cost = project.phases["MonopileInstallation"].port_costs
        monopile_action_cost = monopile_installation_capex - monopile_port_cost
        scour_protection_installation_capex = project.phases["ScourProtectionInstallation"].installation_capex
        scour_protection_port_cost = project.phases["ScourProtectionInstallation"].port_costs
        scour_protection_action_cost = scour_protection_installation_capex - scour_protection_port_cost
    elif "SemiSubmersibleDesign" in project.config["design_phases"]:
        semisubmersible_materials_cost = project.phases["SemiSubmersibleDesign"].total_cost
        mooring_materials_cost = project.phases["MooringSystemDesign"].total_cost
        semisubmersible_installation_capex = project.phases["MooredSubInstallation"].installation_capex
        semisubmersible_port_cost = project.phases["MooredSubInstallation"].port_costs
        semisubmersible_action_cost = semisubmersible_installation_capex - semisubmersible_port_cost
        mooring_installation_capex = project.phases["MooringSystemInstallation"].installation_capex
        mooring_port_cost = project.phases["MooringSystemInstallation"].port_costs
        mooring_action_cost = mooring_installation_capex - mooring_port_cost
    elif "SparDesign" in project.config["design_phases"]:
        spar_materials_cost = project.phases["SparDesign"].total_cost
        mooring_materials_cost = project.phases["MooringSystemDesign"].total_cost
        spar_installation_capex = project.phases["MooredSubInstallation"].installation_capex
        spar_port_cost = project.phases["MooredSubInstallation"].port_costs
        spar_action_cost = spar_installation_capex - spar_port_cost
        mooring_installation_capex = project.phases["MooringSystemInstallation"].installation_capex
        mooring_port_cost = project.phases["MooringSystemInstallation"].port_costs
        mooring_action_cost = mooring_installation_capex - mooring_port_cost
    else:
        raise ValueError(
            f"Unrecognized design_phases for substructure cost lookup: "
            f"{project.config['design_phases']!r}"
        )


    outputs.range("C40").value = monopile_materials_cost
    outputs.range("C43").value = scour_protection_materials_cost
    outputs.range("C46").value = spar_materials_cost
    outputs.range("C49").value = semisubmersible_materials_cost
    outputs.range("C52").value = mooring_materials_cost


    outputs.range("C56").value = project.phases["ArraySystemDesign"].total_cost
    outputs.range("C59").value = project.phases["ExportSystemDesign"].total_cable_cost
    outputs.range("C62").value = project.phases["OffshoreSubstationDesign"].total_cost


    turbine_installation_cost = project.phases["TurbineInstallation"].installation_capex
    turbine_port_cost = project.phases["TurbineInstallation"].port_costs
    turbine_action_cost = turbine_installation_cost - turbine_port_cost

    array_installation_cost = project.phases["ArrayCableInstallation"].installation_capex
    array_port_cost = project.phases["ArrayCableInstallation"].port_costs
    array_action_cost = array_installation_cost - array_port_cost

    export_installation_cost = project.phases["ExportCableInstallation"].installation_capex
    onshore_transmission_rows = installation_logs[
        (installation_logs.phase == "ExportCableInstallation")
        & (installation_logs.action == "Onshore Construction")
    ].cost.values
    onshore_transmission_cost = onshore_transmission_rows[0] if len(onshore_transmission_rows) > 0 else 0.0
    export_installation_cost -= onshore_transmission_cost
    export_port_cost = project.phases["ExportCableInstallation"].port_costs
    export_action_cost = export_installation_cost - export_port_cost

    oss_installation_cost = project.phases["OffshoreSubstationInstallation"].installation_capex
    oss_port_cost = project.phases["OffshoreSubstationInstallation"].port_costs
    oss_action_cost = oss_installation_cost - oss_port_cost


    outputs.range("C66").value = max(monopile_action_cost, spar_action_cost, semisubmersible_action_cost)
    outputs.range("C69").value = mooring_action_cost
    outputs.range("C72").value = turbine_action_cost
    outputs.range("C75").value = array_action_cost
    outputs.range("C78").value = export_action_cost
    outputs.range("C81").value = oss_action_cost
    outputs.range("C84").value = scour_protection_action_cost


    port_costs = [
        [max(monopile_port_cost, spar_port_cost, semisubmersible_port_cost)],
        [mooring_port_cost],
        [turbine_port_cost],
        [array_port_cost],
        [export_port_cost],
        [oss_port_cost],
        [scour_protection_port_cost],
    ]
    outputs.range("C88").value = port_costs


    outputs.range("C102").value = onshore_transmission_cost


    time_outputs = wb.sheets["Installation Process Times"]
    time_outputs.range("A2").expand('table').value = ""
    summary = installation_logs.groupby(["agent", "phase_name", "action"]).sum()["duration"].reset_index()
    time_outputs.range("A2").value = summary.values






def main():
    wb = xw.Book.caller()

    jedi_config = defaultdict(dict)
    jedi_config = dict(read_excel_inputs(wb, jedi_config))
    jedi_config = json.loads(json.dumps(jedi_config))

    array, export = tidy_cable_data(
        jedi_config["array_system_design"], jedi_config["export_system_design"]
    )
    jedi_config["array_system_design"] = array
    jedi_config["export_system_design"] = export
    design_phases, install_phases = determine_phases(wb)
    jedi_config["design_phases"] = design_phases
    jedi_config["install_phases"] = install_phases
    jedi_config = determine_vessels(jedi_config)

    weather = _load_weather(wb.sheets["Project Data - Step 1"].range("D20").value)
    project = ProjectManager(jedi_config, weather=weather, library_path=str(LIBRARY))







    project.run()
    write_outputs(wb, project)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Runs the JEDI-ORBIT script.", prog="jedi-orbit"
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="file_name",
        type=str,
        default="jedi-osw-model.xlsm",
        required=False,
        help=("The JEDI file, a marco-enabled Excel workbook"),
    )
    args = parser.parse_args()

    wb = xw.Book(args.file_name).set_mock_caller()
    main()
