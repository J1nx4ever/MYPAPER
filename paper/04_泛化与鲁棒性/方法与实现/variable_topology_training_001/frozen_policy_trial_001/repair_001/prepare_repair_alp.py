"""Build the independent B repair from the sealed native frozen-policy ALP."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
TRIAL = HERE.parent
sys.path.insert(0, str(TRIAL.parent / 'physical_trip_001'))

from prepare_trip_alp import named_block, replace

BASE_MODEL = 'DRL-PPO-baseline-topologydemo-frozen-policy-trial'
BASE_SHA256 = 'b6411c6eec9bdc994c0455e877ed2db980216b3bdc51b6f284a0971f3ae2ede0'
MODEL = 'DRL-PPO-baseline-topologydemo-frozen-policy-trial-b-repair'
EXPERIMENT = 'Topology_Frozen_Policy_B_Repair'


def patch_dispatch(source):
    for module in ('java.io.IOException', 'java.net.URI', 'java.net.http.HttpClient',
                   'java.net.http.HttpRequest', 'java.net.http.HttpResponse', 'java.time.Duration',
                   'com.google.gson.JsonParser'):
        source = replace(source, f'import {module};\n', '')
    source = replace(source, 'import java.util.ArrayList;', 'import java.nio.file.Path;\nimport java.util.ArrayList;')
    source = replace(source, '    private static final URI ENDPOINT = URI.create("http://127.0.0.1:50871/decide");\n', '')
    source = replace(source, '    private static final Duration TIMEOUT = Duration.ofSeconds(15);\n', '')
    source = replace(source,
        '    private final HttpClient client = HttpClient.newBuilder().connectTimeout(TIMEOUT).build();',
        '    private final TopologyRetryTransport transport;')
    source = replace(source, 'String configSha, String checkpointSha) {',
                     'String configSha, String checkpointSha, Path output) {')
    source = replace(source, '        this.checkpointSha = checkpointSha;',
                     '        this.checkpointSha = checkpointSha;\n        this.transport = new TopologyRetryTransport(output);')
    source = replace(source, 'JsonObject reply = post(body);', 'JsonObject reply = transport.post(body);')
    start = source.index('    private JsonObject post(JsonObject body) {')
    end = source.index('    private static String string(', start)
    report = '''    void recordFailure(String phase, long sequence, Truck truck, Throwable error) {
        transport.recordFailure(phase, main.experimentRunIndex, sequence, truck == null ? -1 : truck.truckId,
                main.time(SECOND), error);
    }

'''
    return source[:start] + report + source[end:]


def wrap_witness_method(source, signature, next_method, phase, truck='truck'):
    start = source.index(signature)
    end = source.index(next_method, start)
    method = source[start:end]
    header, body = method.split('\n', 1)
    if not body.endswith('    }\n\n'):
        raise ValueError(f'Unexpected Java method ending: {signature}')
    body = body.removesuffix('    }\n\n')
    inner = ''.join('    ' + line if line.strip() else line for line in body.splitlines(keepends=True))
    wrapped = (f'{header}\n        try {{\n{inner}        }} catch (RuntimeException | Error error) {{\n'
               f'            recordFailure("{phase}", {truck}, error);\n'
               '            throw error;\n        }\n    }\n\n')
    return source[:start] + wrapped + source[end:]


def patch_witness(source):
    source = replace(source, str(TRIAL / 'configs'), str(HERE / 'configs'))
    source = replace(source, 'new TopologyPolicyDispatch(main, topology, configSha, text("checkpoint_sha256"));',
                     'new TopologyPolicyDispatch(main, topology, configSha, text("checkpoint_sha256"),\n'
                     '                Path.of(text("event_directory")).getParent());')
    source = replace(source, '    public int dispatch(Truck truck) {',
                     '    void recordFailure(String phase, Truck truck, Throwable error) {\n'
                     '        if (dispatcher != null) dispatcher.recordFailure(phase, decisionSequence, truck, error);\n'
                     '    }\n\n    public int dispatch(Truck truck) {')
    methods = (
        ('    public int dispatch(', '    public void applied(', 'dispatch', 'truck'),
        ('    public void applied(', '    private void request(', 'native_reservation_commit', 'truck'),
        ('    public void departing(', '    public void arrived(', 'road_departure', 'truck'),
        ('    public void arrived(', '    public void finish(', 'road_arrival', 'truck'),
        ('    public void finish(', '    public void close(', 'native_shift_close', 'null'),
    )
    for signature, next_method, phase, truck in methods:
        source = wrap_witness_method(source, signature, next_method, phase, truck)
    return source


def patch_dump(source):
    old = '''t.onDumpingFinished(this);
main.fleetResilience.finishDump(t.truckId, time(SECOND));
main.fleetSync();
t.send("restart", t);'''
    return replace(source, old, 'try {\n' + '\n'.join('    ' + line for line in old.splitlines()) +
                   '\n} catch (RuntimeException | Error error) {\n'
                   '    if (main.topologyWitness != null) main.topologyWitness.recordFailure("dump_completion", t, error);\n'
                   '    throw error;\n}')


def prepare_model(base, destination):
    raw = base.read_bytes()
    if sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError('Use the exact sealed frozen-policy trial ALP bytes')
    source = raw.decode('utf-8')
    model = ET.fromstring(source).find('Model')
    if model.findtext('Name') != BASE_MODEL:
        raise ValueError('Use the original frozen-policy trial ALP')
    java_id = max(int(node.text) for node in model.iter('Id') if node.text and node.text.isdecimal()) + 1

    for name, patch in (('TopologyPolicyDispatch', patch_dispatch), ('TopologyWitness', patch_witness)):
        block = named_block(source, 'JavaClass', name)
        source = replace(source, block, patch(block))
    source = patch_dump(source)
    transport = (HERE / 'TopologyRetryTransport.java').read_text(encoding='utf-8')
    source = replace(source, '\t</JavaClasses>',
        f'\t\t<JavaClass>\n\t\t\t<Id>{java_id}</Id>\n'
        '\t\t\t<Name><![CDATA[TopologyRetryTransport]]></Name>\n'
        f'\t\t\t<Text><![CDATA[{transport.removeprefix("package fusion;").lstrip()}]]></Text>\n'
        '\t\t</JavaClass>\n\t</JavaClasses>')

    case_a = named_block(source, 'ParamVariationExperiment', 'Topology_Frozen_Policy_A')
    source = replace(source, case_a, '')
    case_b = named_block(source, 'ParamVariationExperiment', 'Topology_Frozen_Policy_B')
    repair = replace(case_b, 'Topology_Frozen_Policy_B', EXPERIMENT)
    repair = replace(repair, 'Topology frozen Actor 8-hour trial B', 'Topology frozen Actor 8-hour repair B')
    source = replace(source, case_b, repair)
    source = replace(source, f'<Name><![CDATA[{BASE_MODEL}]]>', f'<Name><![CDATA[{MODEL}]]>')

    experiments = ET.fromstring(source).findall('./Model/Experiments/ParamVariationExperiment')
    experiment, = (entry for entry in experiments if entry.findtext('Name') == EXPERIMENT)
    if (experiment.findtext('NumberOfRuns') != '1'
            or experiment.findtext('FreeformParamValue/Expression/Code') != '2'
            or experiment.findtext('./ModelTimeProperties/FinalTime') != '28801.0'):
        raise ValueError('B repair must run one full native shift')
    with destination.open('x', encoding='utf-8') as stream:
        stream.write(source)
    print(sha256(destination.read_bytes()).hexdigest(), destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original_trial_alp', type=Path)
    parser.add_argument('destination_alp', type=Path)
    args = parser.parse_args()
    prepare_model(args.original_trial_alp, args.destination_alp)
