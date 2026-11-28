"""Build an opt-in physical single-trip experiment from the successful preflight ALP."""

import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
MODEL_NAME = 'DRL-PPO-baseline-topologydemo-physical-trip'
EXPERIMENT = 'Topology_Physical_Trip_Check'


def replace(source, old, new, *, count=1):
    if source.count(old) != count:
        raise ValueError(f'expected {count} occurrences of {old!r}')
    return source.replace(old, new)


def named_block(document, tag, name):
    pattern = rf'(?ms)^[\t ]*<{tag}(?: [^>]+)?>.*?^\s*</{tag}>'
    return next(match.group() for match in re.finditer(pattern, document)
                if f'<Name><![CDATA[{name}]]></Name>' in match.group())


def patch_main(main, parameter_id):
    parameter = f"""
				<Variable Class="Parameter">
					<Id>{parameter_id}</Id>
					<Name><![CDATA[topologyTripCase]]></Name>
					<X>700</X><Y>-460</Y>
					<Label><X>10</X><Y>0</Y></Label>
					<PublicFlag>false</PublicFlag>
					<PresentationFlag>true</PresentationFlag>
					<ShowLabel>true</ShowLabel>
					<Properties SaveInSnapshot="true" ModificatorType="STATIC">
						<Type><![CDATA[int]]></Type>
						<UnitType><![CDATA[NONE]]></UnitType>
						<SdArray>false</SdArray>
						<DefaultValue Class="CodeValue"><Code><![CDATA[0]]></Code></DefaultValue>
					</Properties>
				</Variable>"""
    main = replace(main, '</Variables>', parameter + '\n\t\t\t</Variables>')
    main = replace(main, 'FleetResilienceState fleetResilience;',
                   'FleetResilienceState fleetResilience;\nTopologyTripSmoke topologyTripSmoke;')
    main = replace(main, '<StartupCode><![CDATA[boundaryAcceptanceConfigSha256',
                   '<StartupCode><![CDATA[if (topologyTripCase != 0) {\n'
                   '    topologyTripSmoke = TopologyTripSmoke.open(this, topologyTripCase);\n'
                   '    topologyTripSmoke.start();\n    return;\n}\n'
                   'boundaryAcceptanceConfigSha256')
    main = replace(main, '<DestroyCode><![CDATA[if (boundaryEventRecorder',
                   '<DestroyCode><![CDATA[if (topologyTripSmoke != null) {\n'
                   '    topologyTripSmoke.close();\n    return;\n}\n'
                   'if (boundaryEventRecorder')
    main = replace(main, '<Body><![CDATA[int  intDestination = 0;',
                   '<Body><![CDATA[if (topologyTripSmoke != null) return topologyTripSmoke.modelNodeNumber(destination);\n'
                   'int  intDestination = 0;')
    speed_event = named_block(main, 'Event', 'evUpdateSpeedObservation')
    updated = replace(speed_event, '<Action><![CDATA[for (Truck t',
                      '<Action><![CDATA[if (topologyTripSmoke != null) { topologyTripSmoke.sample(); return; }\n'
                      'for (Truck t')
    return replace(main, speed_event, updated)


def patch_truck(truck):
    truck = replace(truck, 'targetLoadSiteId = main.selectNextLoadSite(this);',
                    'targetLoadSiteId = main.topologyTripSmoke == null\n'
                    '        ? main.selectNextLoadSite(this) : main.topologyTripSmoke.dispatch(this);')
    truck = replace(truck, 'destinationNode = main.collectionLoadSites\n        .get(targetLoadSiteId).getName();',
                    'destinationNode = main.topologyTripSmoke == null\n'
                    '        ? main.collectionLoadSites.get(targetLoadSiteId).getName()\n'
                    '        : main.topologyTripSmoke.loadNode(targetLoadSiteId);')
    truck = replace(truck, 'destinationNode = main.collectionDumpSites\n    .get(targetDumpSiteId).getName();',
                    'destinationNode = main.topologyTripSmoke == null\n'
                    '    ? main.collectionDumpSites.get(targetDumpSiteId).getName()\n'
                    '    : main.topologyTripSmoke.dumpNode(targetDumpSiteId);')
    truck = replace(truck, 'moveTo(node);',
                    'if (main.topologyTripSmoke != null) main.topologyTripSmoke.departing(this, node);\n'
                    'moveTo(node);', count=2)
    for name in ('transition18', 'transition24'):
        arrival = named_block(truck, 'StatechartElement', name)
        updated = replace(arrival, 'Trigger="arrival">',
                          'Trigger="arrival">\n\t\t\t\t\t\t<Action><![CDATA['
                          'if (main.topologyTripSmoke != null) main.topologyTripSmoke.arrived(this);]]></Action>')
        truck = replace(truck, arrival, updated)
    truck = replace(truck, '    main.completedTrips++;\n}]]></EntryAction>',
                    '    main.completedTrips++;\n}\n'
                    'if (main.topologyTripSmoke != null) main.topologyTripSmoke.finish(this);]]></EntryAction>')
    transition = named_block(truck, 'StatechartElement', 'transition15')
    updated = replace(transition, '<Condition><![CDATA[true]]></Condition>',
                      '<Condition><![CDATA[main.topologyTripSmoke == null]]></Condition>')
    return replace(truck, transition, updated)


def patch_experiment(source, exp_id, main_id, parameter_id):
    template = named_block(source, 'ParamVariationExperiment', 'Topology_Scene_Preflight')
    template = re.sub(r'ActiveObjectClassId="\d+"', f'ActiveObjectClassId="{main_id}"', template, count=1)
    template = re.sub(r'<Id>\d+</Id>', f'<Id>{exp_id}</Id>', template, count=1)
    template = replace(template, '<Name><![CDATA[Topology_Scene_Preflight]]></Name>',
                       f'<Name><![CDATA[{EXPERIMENT}]]></Name>')
    template = replace(template, 'Topology scene startup preflight', 'Topology production single-trip check')
    template = replace(template, '</NumberOfRuns>', f"""</NumberOfRuns>
			<FreeformParamValue>
				<Id>{parameter_id}</Id>
				<Expression Class="CodeValue">
					<Code><![CDATA[TopologyTripSmoke.caseNumber((int) index)]]></Code>
				</Expression>
			</FreeformParamValue>
			<RangeVariationParamValue>
				<Id>{parameter_id}</Id>
				<Type>FIXED</Type>
			</RangeVariationParamValue>""")
    template = re.sub(r'\n\s*<(?:Freeform|RangeVariation)ParamValue>\s*<Id>(?!' + str(parameter_id) +
                      r'</Id>)\d+</Id>.*?</(?:Freeform|RangeVariation)ParamValue>', '', template, flags=re.S)
    template = re.sub(r'<FinalTime>.*?</FinalTime>', '<FinalTime><![CDATA[2400.0]]></FinalTime>', template)
    return re.sub(r'<FinalDate>.*?</FinalDate>', '<FinalDate><![CDATA[1776775200000]]></FinalDate>', template)


def prepare(path, destination):
    original = path.read_text(encoding='utf-8')
    start_model = ET.fromstring(original).find('Model')
    if start_model.findtext('Name') != 'DRL-PPO-baseline-topologydemo-training-topology':
        raise ValueError('incorrect base model')
    agents = {item.findtext('Name'): item for item in start_model.findall('./ActiveObjectClasses/ActiveObjectClass')}
    main = named_block(original, 'ActiveObjectClass', 'Main')
    truck = named_block(original, 'ActiveObjectClass', 'Truck')
    new_id = max(int(item.text) for item in start_model.iter('Id') if item.text and item.text.isdecimal()) + 1
    result = replace(original, main, patch_main(main, new_id))
    result = replace(result, truck, patch_truck(truck))
    result = replace(result, '\t</Experiments>', patch_experiment(original, new_id + 1,
                     agents['Main'].findtext('Id'), new_id) + '\n\t</Experiments>')
    java = (HERE / 'TopologyTripSmoke.java').read_text().removeprefix('package fusion;\n\n')
    entry = f"""
		<JavaClass>
			<Id>{new_id + 2}</Id>
			<Name><![CDATA[TopologyTripSmoke]]></Name>
			<Text><![CDATA[{java}]]></Text>
		</JavaClass>
"""
    result = replace(result, '\t</JavaClasses>', entry + '\t</JavaClasses>')
    result = replace(result, '<Name><![CDATA[DRL-PPO-baseline-topologydemo-training-topology]]></Name>',
                     f'<Name><![CDATA[{MODEL_NAME}]]></Name>')
    model = ET.fromstring(result).find('Model')
    experiment = next(item for item in model.findall('./Experiments/ParamVariationExperiment')
                      if item.findtext('Name') == EXPERIMENT)
    assert experiment.findtext('./FreeformParamValue/Id') == str(new_id)
    assert len(experiment.findall('FreeformParamValue')) == 1
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
