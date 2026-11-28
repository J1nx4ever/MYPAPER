"""Create an isolated preview ALP from a saved baseline-topologydemo copy."""

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
MODEL_NAME = 'DRL-PPO-baseline-topologydemo-training-topology'
START = 'topologyDemo = new TopologyVisualDemo(this, configPath, configSha256);'
PREVIEW = """if (TrainingTopologyPreview.selects(configPath)) {
    TrainingTopologyPreview.build(this, configPath, configSha256, () -> add_probeTrucks());
    return;
}
"""


def prepare(source, destination):
    original = source.read_text(encoding='utf-8')
    parsed = ET.fromstring(original).find('Model')
    if parsed.findtext('Name') != 'DRL-PPO-baseline-topologydemo' or original.count(START) != 1:
        raise ValueError('expected the baseline-topologydemo ALP with the validated Simulation A')
    agents = {agent.findtext('Name'): agent for agent in parsed.findall('./ActiveObjectClasses/ActiveObjectClass')}
    root = agents['TopologyVisualMain']
    parameters = {param.findtext('Name'): param.findtext('Id') for param in root.findall('./Variables/Variable')}
    sim_a = next(sim for sim in parsed.findall('./Experiments/SimulationExperiment')
                 if sim.findtext('Name') == 'Simulation_Topology_A')
    if any(sim_a.findtext(f"./Parameters/Parameter[ParameterName='{name}']/ParameterValue/Code") is None
           for name in ('configPath', 'configSha256')) or sim_a.findtext('./ModelTimeProperties/FinalTime') != '80.0':
        raise ValueError('use the snapshot containing the verified Simulation A parameters and 80-second duration')

    new_id = max(int(value.text) for value in parsed.iter('Id') if value.text and value.text.isdecimal()) + 1
    entries = []
    for name, getter in (('configPath', 'reference'), ('configSha256', 'sha')):
        parameter_id = parameters[name]
        entries.append(f'''\t\t\t<FreeformParamValue>
\t\t\t\t<Id>{parameter_id}</Id>
\t\t\t\t<Expression Class="CodeValue">
\t\t\t\t\t<Code><![CDATA[TrainingTopologyPreview.{getter}((int) index)]]></Code>
\t\t\t\t</Expression>
\t\t\t</FreeformParamValue>''')
        entries.append(f'''\t\t\t<RangeVariationParamValue>
\t\t\t\t<Id>{parameter_id}</Id>
\t\t\t\t<Type>FIXED</Type>
\t\t\t</RangeVariationParamValue>''')
    experiment = f'''
\t\t<ParamVariationExperiment ActiveObjectClassId="{root.findtext('Id')}">
\t\t\t<Id>{new_id}</Id>
\t\t\t<Name><![CDATA[Topology_Scene_Preflight]]></Name>
\t\t\t<CommandLineArguments><![CDATA[]]></CommandLineArguments>
\t\t\t<MaximumMemory>512</MaximumMemory>
\t\t\t<RandomNumberGenerationType>fixedSeed</RandomNumberGenerationType>
\t\t\t<CustomGeneratorCode>new Random()</CustomGeneratorCode>
\t\t\t<SeedValue>1</SeedValue>
\t\t\t<SelectionModeForSimultaneousEvents>LIFO</SelectionModeForSimultaneousEvents>
\t\t\t<VmArgs><![CDATA[]]></VmArgs>
\t\t\t<LoadRootFromSnapshot>false</LoadRootFromSnapshot>
\t\t\t<AllowParallelEvaluations>false</AllowParallelEvaluations>
\t\t\t<UseFreeformParameters>true</UseFreeformParameters>
\t\t\t<NumberOfRuns>2</NumberOfRuns>
{chr(10).join(entries)}
\t\t\t<ModelTimeProperties>
\t\t\t\t<StopOption><![CDATA[Stop at specified time]]></StopOption>
\t\t\t\t<InitialDate><![CDATA[1776772800000]]></InitialDate>
\t\t\t\t<InitialTime><![CDATA[0.0]]></InitialTime>
\t\t\t\t<FinalDate><![CDATA[1776772800000]]></FinalDate>
\t\t\t\t<FinalTime><![CDATA[0.0]]></FinalTime>
\t\t\t</ModelTimeProperties>
\t\t\t<PresentationProperties>
\t\t\t\t<Title><![CDATA[Topology scene startup preflight]]></Title>
\t\t\t\t<EnableDeveloperPanel>true</EnableDeveloperPanel>
\t\t\t\t<ShowDeveloperPanelOnStart>true</ShowDeveloperPanelOnStart>
\t\t\t</PresentationProperties>
\t\t\t<ReplicationsProperties>
\t\t\t\t<UseReplication>false</UseReplication>
\t\t\t</ReplicationsProperties>
\t\t</ParamVariationExperiment>
'''
    templates = []
    for i, name in enumerate(('TrainingTopology', 'TrainingTopologyPreview'), start=1):
        java = (HERE / f'{name}.java').read_text().removeprefix('package fusion;\n\n')
        if ']]>' in java:
            raise ValueError('unexpected CDATA terminator')
        templates.append(f'''
\t\t<JavaClass>
\t\t\t<Id>{new_id + i}</Id>
\t\t\t<Name><![CDATA[{name}]]></Name>
\t\t\t<Text><![CDATA[{java}]]></Text>
\t\t</JavaClass>
''')
    updated = original.replace('<Name><![CDATA[DRL-PPO-baseline-topologydemo]]></Name>',
                               f'<Name><![CDATA[{MODEL_NAME}]]></Name>', 1)
    updated = updated.replace(START, PREVIEW + START, 1)
    updated = updated.replace('\t</Experiments>', experiment + '\t</Experiments>', 1)
    updated = updated.replace('\t</JavaClasses>', ''.join(templates) + '\t</JavaClasses>', 1)
    result = ET.fromstring(updated).find('Model')
    if result.findtext('Name') != MODEL_NAME or len(result.findall('./Experiments/ParamVariationExperiment')) != 2:
        raise ValueError('candidate ALP structure differs')
    if not re.search(r'I[Dd]>' + str(new_id) + r'</I[Dd]>', updated):
        raise ValueError('missing preview experiment')
    with destination.open('x', encoding='utf-8') as output:
        output.write(updated)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
